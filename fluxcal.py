"""
Flux calibration for KARP (optional; needs a spectrophotometric standard).

A standard star observed with the same setup (disperser, slit, binning) gives the
sensitivity function, which is then applied to every object with that setup:

  1. The standard's combined spectrum (electrons per pixel, on its reference frame's
     scale) is converted to a count rate per Angstrom, C = Flux / (t * dlam), and
     corrected for atmospheric extinction: C0 = C * 10**(0.4 * k(lam) * X), with t and
     X (mid-exposure airmass) of the reference frame.
  2. The tabulated flux of the standard (calib_data/standards/, from HST CALSPEC) is
     smoothed to the spectrograph's resolution and interpolated onto the same grid.
  3. The sensitivity, s = 2.5 log10(C0 / F_ref) (mag), is fit with a cubic spline in
     wavelength, weighted by the errors, leaving out the standard's line cores and the
     telluric bands (calib_data/masks/standard.txt) and clipping outliers.
  4. A spectrum is calibrated as F = C0 / 10**(s / 2.5) (erg/s/cm^2/A), each with its
     own exposure time and airmass.

The shape is reliable. The absolute level is only as good as the match in slit losses
(seeing, centering) and transparency between the standard and the target.
"""
import json
import os
import re

import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage import gaussian_filter1d

from normalize import _knots, in_regions, load_mask

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_data")
STD_DIR = os.path.join(DATA_DIR, "standards")


def _normalize_name(name):
    """Compare object names loosely: case, spaces and punctuation ignored, BD+28D4211 = BD+28 4211."""
    s = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return re.sub(r"(\d)d(\d)", r"\1\2", s)


def standards_index():
    with open(os.path.join(STD_DIR, "index.json")) as f:
        return json.load(f)


def find_standard(objname):
    """Library name of the standard whose name or alias matches objname, or None."""
    target = _normalize_name(objname)
    for name, entry in standards_index().items():
        if target in {_normalize_name(a) for a in [name] + entry.get("aliases", [])}:
            return name
    return None


def load_standard(spec):
    """
    Tabulated flux of a standard: a library name (calib_data/standards/index.json) or a
    file with columns air wavelength (A), flux (erg/s/cm^2/A). Returns (lam, flux, label).
    """
    index = standards_index()
    path = os.path.join(STD_DIR, index[spec]["file"]) if spec in index else spec
    data = np.loadtxt(path, comments="#", usecols=(0, 1))
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1], os.path.basename(path)


def load_extinction(spec):
    """Extinction curve: "apo" (calib_data/extinction/apo.txt) or a file of air wavelength (A), mag/airmass."""
    path = os.path.join(DATA_DIR, "extinction", "apo.txt") if spec == "apo" else spec
    data = np.loadtxt(path, comments="#", usecols=(0, 1))
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def pixel_width(lam):
    """Width of each pixel in Angstroms (any wavelength order)."""
    return np.abs(np.gradient(lam))


def count_rate(lam, flux, sigma, exptime, airmass, extinction):
    """Extinction-corrected count rate per Angstrom (e-/s/A) and its error."""
    ext_lam, k = extinction
    corr = 10 ** (0.4 * np.interp(lam, ext_lam, k) * airmass) / (exptime * pixel_width(lam))
    return flux * corr, sigma * corr


def reference_on_grid(lam, ref_lam, ref_flux, fwhm):
    """
    Tabulated flux smoothed with a Gaussian of FWHM `fwhm` (A) and interpolated onto lam.
    NaN outside the tabulated range.
    """
    step = min(0.5, fwhm / 6)
    fine = np.arange(ref_lam[0], ref_lam[-1], step)
    smooth = gaussian_filter1d(np.interp(fine, ref_lam, ref_flux), fwhm / 2.3548 / step, mode="nearest")
    return np.interp(lam, fine, smooth, left=np.nan, right=np.nan)


def fit_sensitivity(lam, sens, sens_err, bad, regions, knot_spacing=250.0, clip_sigma=3.0,
                    max_iter=20, min_points=15):
    """
    Spline fit to the sensitivity s(lam) (mag), weighted by 1/error, leaving out `bad`
    pixels and the masked regions, with symmetric sigma clipping. Returns
    (fit at every pixel, used, rms of the used residuals in mag, notes).
    """
    order = np.argsort(lam)
    x, y, e = lam[order], sens[order], sens_err[order]
    ok = ~bad[order] & np.isfinite(y) & np.isfinite(e) & (e > 0) & ~in_regions(x, regions)
    if ok.sum() < 4 * min_points:
        raise RuntimeError(f"Only {ok.sum()} usable pixels for the sensitivity fit")
    used = ok.copy()
    for it in range(max_iter):
        knots = _knots(x[used], knot_spacing, min_points)
        spl = LSQUnivariateSpline(x[used], y[used], knots, w=1 / e[used], k=3)
        fit = spl(x)
        resid = y - fit
        scale = mad_std(resid[used])
        new_used = ok & (np.abs(resid) < clip_sigma * scale)
        if np.array_equal(new_used, used):
            break
        used = new_used
    rms = float(np.sqrt(np.mean(resid[used] ** 2)))
    notes = [f"{len(knots)} interior knots, {used.sum()} of {len(x)} pixels used after {it + 1} "
             f"iterations, RMS about the fit {rms:.4f} mag"]
    out_fit, out_used = np.empty_like(fit), np.empty_like(used)
    out_fit[order], out_used[order] = fit, used
    return out_fit, out_used, rms, notes


def sensitivity(lam, flux, sigma, bad, exptime, airmass, extinction, ref, cfg):
    """
    Sensitivity function from a standard's combined spectrum. `ref` is
    (ref_lam, ref_flux); `cfg` is the [fluxcal] config section. Returns a dict with the
    measured and fitted sensitivity (mag) on lam, the pixels used, the RMS and notes.
    """
    rate, rate_err = count_rate(lam, flux, sigma, exptime, airmass, extinction)
    fwhm = cfg["resolution"] if cfg["resolution"] > 0 else 3 * float(np.median(pixel_width(lam)))
    ref_flux = reference_on_grid(lam, *ref, fwhm)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = 2.5 * np.log10(rate / ref_flux)
        sens_err = 2.5 / np.log(10) * rate_err / rate
    regions = load_mask(os.path.join(DATA_DIR, "masks", "standard.txt") if cfg["mask"] == "default" else cfg["mask"])
    if len(cfg["extra_mask"]):
        regions = np.vstack([regions, np.array(cfg["extra_mask"], dtype=float).reshape(-1, 2)])
    bad = bad | ~np.isfinite(sens) | (rate <= 0)
    fit, used, rms, notes = fit_sensitivity(lam, sens, sens_err, bad, regions,
                                            cfg["knot_spacing"], cfg["clip_sigma"])
    covered = np.isfinite(ref_flux) # the fit is only valid where the standard is tabulated
    lam_used = lam[used]
    return {"lam": lam, "sens": sens, "fit": np.where(covered, fit, np.nan), "used": used, "rms": rms,
            "range": (float(lam_used.min()), float(lam_used.max())), "regions": regions,
            "ref_flux": ref_flux, "fwhm": fwhm, "notes": notes}


def calibrate(lam, flux, sigma, exptime, airmass, extinction, sens_lam, sens_fit, valid_range,
              max_extrapolate=100.0):
    """
    Flux-calibrated spectrum (erg/s/cm^2/A) and error. Inside the wavelength range the
    sensitivity fit used, the fit is interpolated; up to `max_extrapolate` A beyond it,
    it is extended linearly (in mag) with the slope at the end of the range; further
    out the result is NaN. Returns (flux, sigma, calibrated, extrapolated).
    """
    rate, rate_err = count_rate(lam, flux, sigma, exptime, airmass, extinction)
    order = np.argsort(sens_lam)
    sl, sf = sens_lam[order], sens_fit[order]
    ok = np.isfinite(sf)
    sl, sf = sl[ok], sf[ok]
    lo, hi = valid_range
    s = np.interp(lam, sl, sf)
    for edge, sign in ((lo, -1), (hi, +1)):
        # slope over the last 50 A inside the fitted range
        inner = np.clip(edge - sign * 50.0, lo, hi)
        slope = (np.interp(edge, sl, sf) - np.interp(inner, sl, sf)) / (edge - inner)
        beyond = sign * (lam - edge) > 0
        s = np.where(beyond, np.interp(edge, sl, sf) + slope * (lam - edge), s)
    inside = (lam >= lo) & (lam <= hi)
    extrapolated = ~inside & (np.minimum(np.abs(lam - lo), np.abs(lam - hi)) <= max_extrapolate)
    calibrated = inside | extrapolated
    factor = np.where(calibrated, 10 ** (-s / 2.5), np.nan)
    return rate * factor, rate_err * factor, calibrated, extrapolated
