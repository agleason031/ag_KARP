"""
Tracing the spectrum on the 2D frame.

  1. find_coarse_trace: the peak of the median spatial profile in blocks of rows,
     followed outward from the middle of the frame (so it follows any tilt).
  2. fit_trace: a Gaussian + constant fit to each row around the coarse trace, then a
     polynomial fit to the Gaussian centers with iterative 3-sigma clipping. The result
     is a sub-pixel trace for every row (and the same rounded to whole columns).

Rows are the dispersion axis and columns the spatial axis. Sizes given in unbinned
pixels are converted with the frame's binning.
"""
import logging

import numpy as np
from astropy.stats import mad_std
from joblib import Parallel, delayed
from numba import njit
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares
from tqdm import tqdm

from .detector import to_binned

log = logging.getLogger("karp")


@njit
def _gauss(x, a, mu, sigma, bck):
    return (a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))) + bck


@njit
def _residuals(p, x, y):
    res = np.empty_like(y)
    a, mu, sigma, bck = p
    for i in range(len(x)):
        res[i] = _gauss(x[i], a, mu, sigma, bck) - y[i]
    return res


def _fit_row(x_vals, y_vals, y_bck, center):
    """Gaussian + constant fit to one row: [amplitude, center, sigma, background], NaN if it fails."""
    if x_vals is None:
        return [np.nan, np.nan, np.nan, np.nan]
    bck = np.median(y_bck)
    p0 = [np.max(y_vals) - bck, center, 2.0, bck]
    bounds = ([0, 0, 1, -100], [np.inf, np.inf, np.inf, np.inf])
    result = least_squares(_residuals, p0, args=(x_vals, y_vals), bounds=bounds, max_nfev=30)
    return result.x if result.success else [np.nan, np.nan, np.nan, np.nan]


def _row_fit_inputs(image, centers, aperture_half, buffer, background):
    """
    For each row: (x, data) over the aperture around centers[row] and the background
    pixels on both sides beyond the buffer, for _fit_row. Rows whose windows run off
    the frame get (None, None, None, center).
    """
    n_rows, n_cols = image.shape
    clr = np.round(centers).astype(int)
    fit_start, fit_end = clr - aperture_half, clr + aperture_half
    buff_start, buff_end = fit_start - buffer, fit_end + buffer
    bck_start, bck_end = buff_start - background, buff_end + background
    valid = (bck_start >= 0) & (bck_end < n_cols)
    inputs = []
    for i in range(n_rows):
        if not valid[i]:
            inputs.append((None, None, None, clr[i]))
            continue
        row = image[i]
        x_vals = np.arange(fit_start[i], fit_end[i])
        y_bck = np.concatenate([row[bck_start[i]:buff_start[i]], row[buff_end[i]:bck_end[i]]])
        inputs.append((x_vals, row[fit_start[i]:fit_end[i]], y_bck, clr[i]))
    return inputs


def find_coarse_trace(image, binning, start_col=None, block=100, search=20):
    """
    Rough trace column for every row, from the peak of the median spatial profile in
    blocks of `block` rows, starting at the middle of the frame and looking within
    +/- `search` columns of the previous block's peak (block and search in unbinned
    pixels; binning = (binx, biny)). start_col: column to start from (default: the
    brightest column of the middle half of the frame). Blocks with no clear peak are
    interpolated over.
    """
    binx, biny = binning
    block = to_binned(block, biny, minimum=10)
    search = to_binned(search, binx, minimum=2)
    n_rows, n_cols = image.shape
    starts = np.arange(0, n_rows, block)
    centers = starts + block / 2
    profiles = np.array([np.nanmedian(image[s:s + block], axis=0) for s in starts])
    profiles -= np.nanmedian(profiles, axis=1, keepdims=True) # Remove the sky/background level

    if start_col is None:
        col0 = int(np.nanargmax(np.nanmedian(image[n_rows // 4: 3 * n_rows // 4], axis=0)))
    else:
        col0 = start_col

    def peak_near(k, col):
        lo, hi = max(0, col - search), min(n_cols, col + search + 1)
        seg = profiles[k, lo:hi]
        j = int(np.nanargmax(seg))
        # Require a clear peak (5x the profile's robust scatter)
        return lo + j if seg[j] > 5 * mad_std(profiles[k], ignore_nan=True) else None

    peaks = np.full(len(starts), np.nan)
    k_mid = len(starts) // 2
    for direction in (range(k_mid, len(starts)), range(k_mid - 1, -1, -1)):
        col = col0
        for k in direction:
            p = peak_near(k, col)
            if p is not None:
                peaks[k] = p
                col = p

    found = ~np.isnan(peaks)
    if found.sum() < 2:
        raise RuntimeError("KARP could not find the trace; set [trace] col in the config")
    return np.round(np.interp(np.arange(n_rows), centers[found], peaks[found])).astype(int)


def fit_trace(image, aperture_half, buffer, background, order, binning, start_col=None):
    """
    Trace center (column) for every row. aperture_half, buffer, background: aperture half-width, buffer
    and background widths in columns (as for the extraction); order: polynomial order.
    Returns (trace rounded to whole columns, sub-pixel trace, per-row Gaussian fits
    [amplitude, center, sigma, background]).
    """
    n_rows = image.shape[0]
    rows = np.arange(n_rows)
    coarse = find_coarse_trace(image, binning, start_col)

    log.debug("Fitting the trace")
    inputs = _row_fit_inputs(image, coarse, aperture_half, buffer, background)
    fits = np.array(Parallel(n_jobs=-1)(delayed(_fit_row)(*args) for args in tqdm(inputs)))
    a_vals, mu_vals, sig_vals = fits[:, 0], fits[:, 1], fits[:, 2]

    # Usable fits: converged, centered within the aperture, and not too broad
    ok = (np.isfinite(a_vals) & np.isfinite(mu_vals)
          & (np.abs(mu_vals - coarse) < aperture_half) & (sig_vals < aperture_half))
    if ok.sum() <= order:
        raise RuntimeError("Too few rows with a usable Gaussian fit to fit the trace")

    # Polynomial fit with iterative 3-sigma clipping (rejected rows can come back in)
    used = ok.copy()
    for _ in range(10):
        poly = Polynomial.fit(rows[used], mu_vals[used], order)
        resid = mu_vals - poly(rows)
        scatter = mad_std(resid[used])
        new_used = ok & (np.abs(resid) < 3 * scatter)
        if np.array_equal(new_used, used):
            break
        used = new_used

    trace = poly(rows)
    columns = np.round(trace).astype(int)
    log.info(f"Trace: order {order} polynomial, {used.sum()} of {n_rows} rows used, "
             f"residual scatter {scatter:.3f} pix, columns {columns.min()}-{columns.max()}")
    log.debug(f"Trace polynomial coefficients: {poly.convert().coef}")
    return columns, trace, fits
