"""
Spectral extraction for KARP.

For each detector row (the dispersion runs along the rows):
  - The sky is fit with a low-order polynomial across the background windows on
    both sides of the trace (with sigma clipping) and evaluated under the aperture.
  - Boxcar: sum of sky-subtracted data over the aperture, with fractional weights
    for the pixels at the aperture edges (the trace position is sub-pixel).
  - Optimal (Horne 1986, PASP 98, 609): weight pixels by the spatial profile over
    the variance. The profile is measured from the data (sky-subtracted fraction of
    the flux in each pixel) and smoothed along the rows; the variance comes from the
    model (profile x flux + sky), and pixels that deviate by more than reject_sigma
    are left out one at a time.
Both return the flux, its variance (including the uncertainty of the sky fit), and
the sky under the aperture, all in electrons.

Geometry, in pixels, relative to the trace center t of each row:
  aperture:    |x - t| <= half                    (half = appw + 0.5, i.e. 2*appw+1 pixels)
  background:  half + buffw <= |x - t| <= half + buffw + bckw
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


def variance_image(sci, flat, readnoise):
    """
    Per-pixel variance (e-^2) of a bias-subtracted, flat-fielded frame in electrons.
    Before flat-fielding a pixel had sci*flat electrons with Poisson + read noise;
    dividing by the flat divides the variance by flat^2.
    """
    raw = sci * flat
    return (np.maximum(raw, 0.0) + readnoise ** 2) / flat ** 2


def edge_weights(center, half, x):
    """Fraction of each pixel (centered on x, width 1) inside [center - half, center + half]."""
    return np.clip(np.minimum(x + 0.5, center + half) - np.maximum(x - 0.5, center - half), 0.0, 1.0)


def _fit_sky(x, d, order, clip_sigma=3.0, max_iter=5):
    """
    Polynomial (in x) fit to background pixels with sigma clipping.
    Returns (coefficients, covariance of the coefficients) or (None, None).
    The covariance uses the empirical scatter of the residuals.
    """
    use = np.isfinite(d)
    A = np.vander(x, order + 1)
    for _ in range(max_iter):
        if use.sum() <= order + 1:
            return None, None
        c, *_ = np.linalg.lstsq(A[use], d[use], rcond=None)
        resid = d - A @ c
        s2 = np.sum(resid[use] ** 2) / (use.sum() - order - 1)
        new_use = np.isfinite(d) & (np.abs(resid) <= clip_sigma * np.sqrt(s2))
        if np.array_equal(new_use, use):
            break
        use = new_use
    cov = s2 * np.linalg.inv(A[use].T @ A[use])
    return c, cov


def _smooth_profile(frac, window):
    """NaN-aware running mean along the rows (axis 0) of a 2D array."""
    good = np.isfinite(frac)
    num = uniform_filter1d(np.where(good, frac, 0.0), size=window, axis=0, mode="nearest")
    den = uniform_filter1d(good.astype(float), size=window, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def extract(sci, var, flat, trace, appw, buffw, bckw, readnoise, method="boxcar",
            sky_order=1, profile_window=101, reject_sigma=5.0, diag_rows=()):
    """
    Extract a spectrum from a 2D frame (electrons, bias-subtracted, flat-fielded).

    sci, var, flat: 2D arrays (rows x columns); trace: sub-pixel trace center per row.
    Returns a dict of per-row arrays: flux, var, sky (electrons under the aperture),
    n_rejected (optimal only), and "diag": {row: details} for the rows in diag_rows.
    Rows where the aperture or background falls off the frame get NaN.
    """
    n_rows, n_cols = sci.shape
    half = appw + 0.5
    inner, outer = half + buffw, half + buffw + bckw
    flux = np.full(n_rows, np.nan)
    fvar = np.full(n_rows, np.nan)
    sky_tot = np.full(n_rows, np.nan)
    n_rej = np.zeros(n_rows, dtype=int)
    diag = {}

    # Pass 1: sky fit and boxcar flux for every row
    lo_all = np.floor(trace - outer).astype(int)
    hi_all = np.ceil(trace + outer).astype(int) + 1
    rows_ok = (lo_all >= 0) & (hi_all <= n_cols) & np.isfinite(trace)
    sky_fits = {}
    for r in np.nonzero(rows_ok)[0]:
        t = trace[r]
        x = np.arange(lo_all[r], hi_all[r])
        d = sci[r, x]
        dist = np.abs(x - t)
        bkg = (dist >= inner) & (dist <= outer)
        c, cov = _fit_sky(x[bkg] - t, d[bkg], sky_order)
        if c is None:
            continue
        w = edge_weights(t, half, x)
        ap = w > 0
        B = np.vander(x[ap] - t, sky_order + 1)
        sky = B @ c
        u = w[ap]
        flux[r] = np.sum(u * (d[ap] - sky))
        fvar[r] = np.sum(u ** 2 * var[r, x[ap]]) + u @ B @ cov @ B.T @ u
        sky_tot[r] = np.sum(u * sky)
        sky_fits[r] = (x[ap], sky, B, cov, u)
        if r in diag_rows:
            diag[r] = {"x": x, "data": d, "sky": np.vander(x - t, sky_order + 1) @ c, "trace": t,
                       "half": half, "inner": inner, "outer": outer, "boxcar": flux[r]}

    if method == "boxcar":
        return {"flux": flux, "var": fvar, "sky": sky_tot, "n_rejected": n_rej, "diag": diag}
    if method != "optimal":
        raise ValueError(f"Unknown extraction method {method!r}")

    # Pass 2: spatial profile from the data. For each row, the sky-subtracted fraction of
    # the boxcar flux in each detector column, smoothed along the rows per column.
    good_rows = np.array(sorted(sky_fits))
    cmin = min(sky_fits[r][0][0] for r in good_rows)
    cmax = max(sky_fits[r][0][-1] for r in good_rows)
    frac = np.full((n_rows, cmax - cmin + 1), np.nan)
    for r in good_rows:
        xa, sky, *_ = sky_fits[r]
        if flux[r] > 0:
            frac[r, xa - cmin] = (sci[r, xa] - sky) / flux[r]
    profile = _smooth_profile(frac, profile_window)

    # Pass 3: optimal flux per row with model variance and outlier rejection
    for r in good_rows:
        xa, sky, B, cov, u = sky_fits[r]
        P = np.maximum(profile[r, xa - cmin], 0.0)
        if not np.all(np.isfinite(P)) or P.sum() <= 0:
            continue
        P = P / P.sum()
        d = sci[r, xa] - sky
        f_row = flat[r, xa]
        use = np.ones(len(xa), dtype=bool)
        f = flux[r] if np.isfinite(flux[r]) else np.sum(d)
        for _ in range(len(xa)):
            V = (np.maximum((f * P + sky) * f_row, 0.0) + readnoise ** 2) / f_row ** 2
            den = np.sum(P[use] ** 2 / V[use])
            f = np.sum(P[use] * d[use] / V[use]) / den
            z = np.where(use, (d - f * P) ** 2 / V, 0.0)
            worst = int(np.argmax(z))
            if z[worst] <= reject_sigma ** 2 or use.sum() <= 3:
                break
            use[worst] = False
        V = (np.maximum((f * P + sky) * f_row, 0.0) + readnoise ** 2) / f_row ** 2
        wts = np.where(use, P / V, 0.0) / np.sum(P[use] ** 2 / V[use])
        flux[r] = f
        fvar[r] = 1.0 / np.sum(P[use] ** 2 / V[use]) + wts @ B @ cov @ B.T @ wts
        n_rej[r] = int((~use).sum())
        if r in diag:
            diag[r]["optimal"] = f
            diag[r]["model_x"] = xa
            diag[r]["model"] = sky + f * P
    return {"flux": flux, "var": fvar, "sky": sky_tot, "n_rejected": n_rej, "diag": diag}
