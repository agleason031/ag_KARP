"""
Flux calibration for KARP (optional; needs a spectrophotometric standard).

A standard star observed with the same setup (disperser, slit, binning) gives the
sensitivity function, which is then applied to every object with that setup:

  1. The standard's combined spectrum (electrons per pixel, on its reference frame's
     scale) is converted to a count rate per Angstrom, C = Flux / (t * dlam), and
     corrected for atmospheric extinction: C0 = C * 10**(0.4 * k(lam) * X), with t and
     X (mid-exposure airmass) of the reference frame.
  2. The tabulated flux of the standard (karp/data/standards/, from HST CALSPEC) and
     the observed count rate are brought to a common resolution: at each wavelength,
     whichever is sharper is smoothed to the resolution of the other (CALSPEC's optical
     spectra, 5.5-9.8 A FWHM, are usually the coarser). The KOSMOS resolution is the
     arc line width unless set; the result depends little on it.
  3. A wavelength offset between the standard and its tabulated spectrum (flexure,
     calibration error) is fit by lining up the lines, and taken out for this
     comparison only (it is logged as a wavelength check).
  4. The sensitivity, s = 2.5 log10(C0 / F_ref) (mag), is fit with a cubic spline in
     wavelength, weighted by the errors, leaving out the standard's line cores and the
     telluric bands (karp/data/masks/standard.txt) and clipping outliers.
  5. A spectrum is calibrated as F = C0 / 10**(s / 2.5) (erg/s/cm^2/A), each with its
     own exposure time and airmass. Science spectra are never smoothed.

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

from .regions import in_regions, mask_regions, spline_knots

from .paths import data_file

STD_DIR = data_file("standards")
# Resolution (FWHM, A) of CALSPEC's optical STIS spectra (G430L below ~5700 A, G750L
# above), used when a file gives none
STIS_FWHM = ((5700.0, 5.5), (np.inf, 9.8))
STEP = 0.25 # A, fine grid for smoothing


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
    Tabulated flux of a standard: a library name (karp/data/standards/index.json) or a
    file with columns air wavelength (A), flux (erg/s/cm^2/A) and optionally its
    resolution (FWHM, A). Returns (lam, flux, fwhm, label); a missing or zero FWHM is
    taken as that of CALSPEC's STIS spectra (STIS_FWHM).
    """
    index = standards_index()
    path = os.path.join(STD_DIR, index[spec]["file"]) if spec in index else spec
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    data = data[np.argsort(data[:, 0])]
    lam, flux = data[:, 0], data[:, 1]
    fwhm = data[:, 2] if data.shape[1] > 2 else np.zeros_like(lam)
    default = np.select([lam < edge for edge, _ in STIS_FWHM], [f for _, f in STIS_FWHM])
    return lam, flux, np.where(fwhm > 0, fwhm, default), os.path.basename(path)


def load_extinction(spec):
    """Extinction curve: "apo" (karp/data/extinction/apo.txt) or a file of air wavelength (A), mag/airmass."""
    path = data_file("extinction", "apo.txt") if spec == "apo" else spec
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


def smooth_variable(y, fwhm, step=STEP):
    """
    Gaussian smoothing of y (on a uniform grid of `step` A) with a FWHM (A) that varies
    along the grid; FWHM below one step leaves y unchanged. Widths are rounded to steps
    of 0.25 A, and each distinct width is applied with one convolution.
    """
    q = np.round(np.asarray(fwhm) / 0.25) * 0.25
    out = np.array(y, dtype=float)
    for width in np.unique(q[q >= step]):
        sel = q == width
        out[sel] = gaussian_filter1d(y, width / 2.3548 / step, mode="nearest")[sel]
    return out


def common_resolution(lam, rate, ref, inst_fwhm, offset=0.0):
    """
    Observed count rate and tabulated flux at a common resolution, on lam. ref is
    (ref_lam, ref_flux, ref_fwhm); inst_fwhm is the spectrograph's resolution (A);
    offset (A) is how far the observed wavelengths are from the tabulated ones
    (observed - true), so the tabulated spectrum is moved by +offset. At each wavelength
    the sharper of the two is smoothed to the other's resolution. Returns
    (rate, ref_flux), NaN outside the tabulated range.
    """
    ref_lam, ref_flux, ref_fwhm = ref
    order = np.argsort(lam)
    x, y = lam[order], rate[order]
    good = np.isfinite(y)
    fine = np.arange(x[0], x[-1] + STEP, STEP)
    obs = np.interp(fine, x[good], y[good])
    tab = np.interp(fine, ref_lam + offset, ref_flux, left=np.nan, right=np.nan)
    tab_fwhm = np.interp(fine, ref_lam + offset, ref_fwhm)
    target = np.maximum(inst_fwhm, tab_fwhm)
    obs = smooth_variable(obs, np.sqrt(target ** 2 - inst_fwhm ** 2))
    inside = np.isfinite(tab)
    tab[inside] = smooth_variable(tab[inside], np.sqrt(target ** 2 - tab_fwhm ** 2)[inside])
    out_rate, out_ref = np.empty(len(lam)), np.empty(len(lam))
    out_rate[order] = np.interp(x, fine, obs)
    out_ref[order] = np.interp(x, fine[inside], tab[inside], left=np.nan, right=np.nan)
    return out_rate, out_ref


def line_mismatch(lam, rate, ref, inst_fwhm, offset, regions, knot_spacing):
    """
    RMS (mag) of log(rate / ref), about a smooth spline through the unmasked pixels,
    inside the masked regions (the standard's lines): how badly the lines line up.
    """
    r, f = common_resolution(lam, rate, ref, inst_fwhm, offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 2.5 * np.log10(r / f)
    order = np.argsort(lam)
    x, y = lam[order], s[order]
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    masked = in_regions(x, regions)
    knots = spline_knots(x[~masked], knot_spacing, 15)
    resid = y - LSQUnivariateSpline(x[~masked], y[~masked], knots, k=3)(x)
    return float(np.sqrt(np.mean(resid[masked] ** 2))) if masked.any() else np.nan


def fit_offset(lam, rate, ref, inst_fwhm, regions, knot_spacing, max_offset=5.0):
    """
    Wavelength offset (A, observed - tabulated) that best lines up the standard's lines
    with its tabulated spectrum: a grid search in 0.25 A steps, then 0.05 A steps.
    Returns (offset, at_limit).
    """
    def best(grid):
        scores = [line_mismatch(lam, rate, ref, inst_fwhm, o, regions, knot_spacing) for o in grid]
        return float(grid[int(np.nanargmin(scores))])
    coarse = best(np.arange(-max_offset, max_offset + 1e-9, 0.25))
    fine = best(np.arange(coarse - 0.25, coarse + 0.25 + 1e-9, 0.05))
    return fine, abs(coarse) >= max_offset - 1e-9


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
        knots = spline_knots(x[used], knot_spacing, min_points)
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


def sensitivity(lam, flux, sigma, bad, exptime, airmass, extinction, ref, inst_fwhm, cfg):
    """
    Sensitivity function from a standard's combined spectrum. `ref` is
    (ref_lam, ref_flux, ref_fwhm); inst_fwhm is the spectrograph's resolution (A);
    `cfg` is the [fluxcal] config section. Returns a dict with the measured and fitted
    sensitivity (mag) on lam, the pixels used, the RMS, the wavelength offset and notes.
    """
    rate, rate_err = count_rate(lam, flux, sigma, exptime, airmass, extinction)
    regions = mask_regions(cfg["mask"], cfg["extra_mask"], "standard.txt")
    notes = []
    offset = 0.0
    if cfg["fit_offset"]:
        offset, at_limit = fit_offset(lam, rate, ref, inst_fwhm, regions, cfg["knot_spacing"], cfg["max_offset"])
        if at_limit:
            notes.append(f"WARNING: wavelength offset at the search limit ({cfg['max_offset']:g} A); "
                         "check the wavelength calibration")
    rate_c, ref_flux = common_resolution(lam, rate, ref, inst_fwhm, offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = 2.5 * np.log10(rate_c / ref_flux)
        sens_err = 2.5 / np.log(10) * rate_err / rate
    bad = bad | ~np.isfinite(sens) | (rate <= 0)
    fit, used, rms, fit_notes = fit_sensitivity(lam, sens, sens_err, bad, regions,
                                                cfg["knot_spacing"], cfg["clip_sigma"])
    notes = fit_notes + notes
    covered = np.isfinite(ref_flux) # the fit is only valid where the standard is tabulated
    lam_used = lam[used]
    return {"lam": lam, "sens": sens, "fit": np.where(covered, fit, np.nan), "used": used, "rms": rms,
            "range": (float(lam_used.min()), float(lam_used.max())), "regions": regions,
            "ref_flux": ref_flux, "inst_fwhm": inst_fwhm, "offset": offset, "notes": notes}


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
