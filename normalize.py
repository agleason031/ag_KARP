"""
Continuum normalization for KARP (optional).

The continuum is a cubic spline in wavelength with knots every `knot_spacing` A,
fit with weights 1/sigma to the pixels that are:
  - valid (finite flux and error, not flagged),
  - outside the masked regions (a mask file of [center, half-width] in A, plus any
    extra [lam_min, lam_max] ranges from the config),
  - not rejected by asymmetric sigma clipping: after each fit, pixels more than
    low_reject sigma BELOW the continuum (absorption lines) or high_reject sigma
    ABOVE it (emission, cosmic rays) are left out of the next fit. Sigma is the
    robust scatter of (flux - continuum) / continuum among the pixels in use.
Knots in stretches with too few usable pixels (e.g. inside a broad masked line)
are dropped, so the spline bridges the gap.

Nothing is removed from the spectrum: every pixel gets a continuum value, and the
output records which pixels the fit used.
"""
import os

import numpy as np
from astropy.stats import mad_std
from scipy.interpolate import LSQUnivariateSpline

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_data")


def load_mask(spec):
    """
    Masked regions as an (n, 2) array of [lam_min, lam_max].
    spec: "default" (calib_data/masks/stellar.txt), "none", or a path to a file with
    columns center, half-width (A); anything after the second column is ignored.
    """
    if spec == "none":
        return np.empty((0, 2))
    path = os.path.join(DATA_DIR, "masks", "stellar.txt") if spec == "default" else spec
    regions = []
    with open(path) as f:
        for line in f:
            parts = line.split("#", 1)[0].split()
            if len(parts) >= 2:
                c, hw = float(parts[0]), float(parts[1])
                regions.append([c - hw, c + hw])
    return np.array(regions).reshape(-1, 2)


def in_regions(lam, regions):
    out = np.zeros(len(lam), dtype=bool)
    for lo, hi in regions:
        out |= (lam >= lo) & (lam <= hi)
    return out


def _knots(x, spacing, min_points):
    """Interior knots every `spacing` in x, dropping any that leave fewer than min_points between knots."""
    lo, hi = x[0], x[-1]
    n = int((hi - lo) // spacing)
    if n < 1:
        return np.array([])
    cand = lo + (hi - lo) * np.arange(1, n + 1) / (n + 1)
    knots = []
    last = lo
    for k in cand:
        if np.sum((x > last) & (x <= k)) >= min_points:
            knots.append(k)
            last = k
    # The last interval also needs enough points
    while knots and np.sum(x > knots[-1]) < min_points:
        knots.pop()
    return np.array(knots)


def fit_continuum(lam, flux, sigma, bad, regions, knot_spacing=150.0, low_reject=2.0,
                  high_reject=3.0, max_iter=30, min_points=20):
    """
    Continuum for one spectrum. All arrays are per pixel (any wavelength order).
    bad: pixels never to use. Returns (continuum, used, notes), where `used` marks
    the pixels in the final fit.
    """
    order = np.argsort(lam)
    x, y, s = lam[order], flux[order], sigma[order]
    ok = ~bad[order] & np.isfinite(y) & np.isfinite(s) & (s > 0) & ~in_regions(x, regions)
    if ok.sum() < 4 * min_points:
        raise RuntimeError(f"Only {ok.sum()} usable pixels for the continuum fit")
    used = ok.copy()
    notes = []
    converged = False
    for it in range(max_iter):
        knots = _knots(x[used], knot_spacing, min_points)
        spl = LSQUnivariateSpline(x[used], y[used], knots, w=1 / s[used], k=3)
        cont = spl(x)
        r = (y - cont) / cont
        scale = mad_std(r[used])
        new_used = ok & (r > -low_reject * scale) & (r < high_reject * scale)
        n_changed = int(np.sum(new_used != used))
        if n_changed == 0:
            converged = True
            break
        used = new_used
    status = (f"converged after {it + 1} iterations" if converged else
              f"stopped at {max_iter} iterations with {n_changed} pixels still changing")
    notes.append(f"{len(knots)} interior knots, {status}, {used.sum()} of {len(x)} pixels used, "
                 f"scatter about the continuum {scale * 100:.2f}%")
    out_cont = np.empty_like(cont)
    out_used = np.empty_like(used)
    out_cont[order] = cont
    out_used[order] = used
    return out_cont, out_used, notes
