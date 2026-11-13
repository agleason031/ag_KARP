"""
Combining the extracted spectra of an object's frames.

Each frame is interpolated onto the wavelength grid of the frame with the highest
S/N (the reference) and scaled to it: by a smooth function of wavelength (scale =
"smooth", a polynomial of order scale_order fit to the flux ratio with clipping), a
single factor ("median"), or not at all ("none"). The scaled frames are then
combined pixel by pixel:
  "weighted": inverse-variance weighted mean, after rejecting values more than
              reject_sigma (their own error) from the median of the frames (only
              where >= 3 frames have data); exact error 1/sqrt(sum 1/sigma^2).
              Mask bit 4 marks pixels where a value was rejected.
  "mean":     plain mean; error sqrt(sum sigma^2) / n.
  "median":   median; error that of the inverse-variance mean times the median/mean
              noise factor for that many frames.
Mask bit 2 marks pixels with no data. Flux stays in electrons per pixel, on the
reference frame's scale.
"""
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import interp1d

log = logging.getLogger("karp")

# Error of the median of n values relative to the error of their mean, for Gaussian
# noise and similar errors in each frame (Monte Carlo, 400k trials; large-n limit sqrt(pi/2))
_MEDIAN_ERROR_FACTOR = np.array([1.0, 1.0, 1.0, 1.159, 1.094, 1.196, 1.135, 1.213, 1.161,
                                 1.222, 1.174, 1.232, 1.185, 1.234, 1.194, 1.235])

METHOD_NAMES = {"median": "median", "mean": "plain mean", "weighted": "inverse-variance weighted mean"}


def median_error_factor(n):
    """Factor for each n (array); n = 0 gives 1 (those pixels have no data anyway)."""
    n = np.asarray(n)
    return np.where(n < len(_MEDIAN_ERROR_FACTOR),
                    _MEDIAN_ERROR_FACTOR[np.minimum(n, len(_MEDIAN_ERROR_FACTOR) - 1)],
                    np.sqrt(np.pi / 2))


@dataclass
class Combined:
    """A combined spectrum and how it was made."""
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    ref_frame: int # frames are on this frame's wavelength grid and flux scale
    frames: list = field(default_factory=list)
    n_rejected: dict = field(default_factory=dict) # frame -> pixels rejected ("weighted")


def snr_in(wave, flux, sigma, mask, lam_min, lam_max):
    """Median Flux/Sigma of pixels with data in a wavelength range (NaN if none)."""
    m = (wave >= lam_min) & (wave <= lam_max) & ((mask & 2) == 0)
    return float(np.median(flux[m] / sigma[m])) if m.any() else np.nan


def scale_to_reference(frames, scale="smooth", scale_order=2):
    """
    Put frames {n: (lam, flux, sigma)} on the reference frame's grid and flux scale.
    Returns (reference frame, wavelength grid, {n: (flux, sigma)}).
    """
    frame_snr = {n: float(np.nanmedian(f / s)) for n, (_, f, s) in frames.items()}
    ref = max(frame_snr, key=frame_snr.get)
    wave = frames[ref][0]
    log.info("Frame S/N per pixel (median): " + ", ".join(f"{n}: {v:.1f}" for n, v in frame_snr.items())
             + f"; reference frame {ref}")

    on_grid = {}
    for n, (lam, f, s) in frames.items():
        on_grid[n] = (interp1d(lam, f, bounds_error=False, fill_value=np.nan)(wave),
                      interp1d(lam, s, bounds_error=False, fill_value=np.nan)(wave))

    ref_f = on_grid[ref][0]
    x = (wave - wave.mean()) / (np.ptp(wave) / 2) # scaled wavelength for a well-conditioned fit
    for n, (fi, si) in on_grid.items():
        if n == ref or scale == "none":
            continue
        ratio = ref_f / fi
        ok = np.isfinite(ratio) & (fi > 0) & (ref_f > 0)
        if scale == "median":
            factor = np.full(len(wave), np.median(ratio[ok]))
        else: # smooth
            use = ok.copy()
            for _ in range(10):
                coef = np.polyfit(x[use], ratio[use], scale_order)
                resid = ratio - np.polyval(coef, x)
                new_use = ok & (np.abs(resid) < 3 * mad_std(resid[use]))
                if np.array_equal(new_use, use):
                    break
                use = new_use
            factor = np.polyval(coef, x)
        on_grid[n] = (fi * factor, si * factor)
        log.info(f"Frame {n} scaled to frame {ref}: factor {np.nanmedian(factor):.4f} (median), "
                 f"range {np.nanmin(factor):.4f}-{np.nanmax(factor):.4f}")
    return ref, wave, on_grid


def combine_on_grid(on_grid, nums, method="weighted", reject_sigma=5.0):
    """
    Combine frames `nums` of on_grid {n: (flux, sigma)} pixel by pixel.
    Returns (flux, sigma, mask, pixels rejected per frame).
    """
    gflux = np.array([on_grid[n][0] for n in nums])
    gsig = np.array([on_grid[n][1] for n in nums])
    good = np.isfinite(gflux) & np.isfinite(gsig) & (gsig > 0)
    gflux = np.where(good, gflux, np.nan)
    gsig = np.where(good, gsig, np.nan)
    n_used = good.sum(axis=0)
    n_rej = np.zeros(len(nums), dtype=int)
    mask = np.zeros(gflux.shape[1], dtype=np.int16)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning) # All-NaN columns are flagged below
        med = np.nanmedian(gflux, axis=0)
        if method == "median":
            # Error: that of the inverse-variance mean of the frames used, times how
            # much noisier a median of that many values is than a mean
            inv_var = np.where(good, 1.0 / gsig**2, 0.0)
            flux = med
            sig = np.sqrt(1.0 / np.sum(inv_var, axis=0)) * median_error_factor(n_used)
        elif method == "mean":
            flux = np.nanmean(gflux, axis=0)
            sig = np.sqrt(np.nansum(gsig**2, axis=0)) / n_used
        else: # weighted
            # Reject values more than reject_sigma (their own error) from the median,
            # where >= 3 frames have data (with 2, an outlier drags the median halfway
            # and both would go). Never reject every value at a pixel.
            use = good.copy()
            if reject_sigma > 0: # 0 = no rejection
                dev = np.abs(gflux - med) / gsig
                reject = good & (dev > reject_sigma) & (n_used >= 3)
                reject &= ~np.all(reject | ~good, axis=0) # keep a pixel's values if all would go
                use &= ~reject
                n_rej = reject.sum(axis=1)
                mask[reject.any(axis=0)] |= 4 # 4 = a frame was rejected here (informational)
            inv_var = np.where(use, 1.0 / gsig**2, 0.0)
            wsum = np.sum(inv_var, axis=0)
            flux = np.sum(np.where(use, gflux, 0.0) * inv_var, axis=0) / wsum
            sig = np.sqrt(1.0 / wsum)
    no_data = (n_used == 0) | ~np.isfinite(flux) | ~np.isfinite(sig)
    flux = np.where(no_data, np.nan, flux)
    mask[no_data] |= 2 # 2 = no data
    return flux, sig, mask, n_rej


def combine_frames(frames, scale="smooth", scale_order=2, method="weighted", reject_sigma=5.0,
                   jackknife=False):
    """
    Combine extracted spectra {n: (lam, flux, sigma)} (see the module docstring).
    With jackknife=True, also log the S/N at 5500 A with each frame left out in turn.
    Returns a Combined.
    """
    ref, wave, on_grid = scale_to_reference(frames, scale, scale_order)
    nums = list(frames)
    if jackknife:
        for leave in nums:
            f, s, m, _ = combine_on_grid(on_grid, [n for n in nums if n != leave], method, reject_sigma)
            log.info(f"S/N at 5500 A: {snr_in(wave, f, s, m, 5450, 5550):.1f} without frame {leave}")

    flux, sigma, mask, n_rej = combine_on_grid(on_grid, nums, method, reject_sigma)
    desc = METHOD_NAMES[method]
    if method == "weighted":
        desc += f", {reject_sigma:g} sigma rejection" if reject_sigma > 0 else ", no rejection"
    log.info(f"Combined {len(nums)} frames: {desc}")
    if method == "weighted":
        log.info("Pixels rejected per frame: " + ", ".join(f"{n}: {r}" for n, r in zip(nums, n_rej)))
    return Combined(wave, flux, sigma, mask, ref, nums, dict(zip(nums, (int(r) for r in n_rej))))
