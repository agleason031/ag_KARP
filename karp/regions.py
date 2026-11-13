"""
Wavelength regions and spline knots shared by continuum normalization and flux
calibration.

Mask files (karp/data/masks/) list regions to leave out of a fit as center and
half-width in Angstroms (air), one per line; anything after the second column is a
comment.
"""
import numpy as np

from .paths import data_file


def load_mask(spec, default):
    """
    Masked regions as an (n, 2) array of [lam_min, lam_max]. spec: "default" (the file
    `default` in karp/data/masks/), "none", or the path of a mask file.
    """
    if spec == "none":
        return np.empty((0, 2))
    path = data_file("masks", default) if spec == "default" else spec
    regions = []
    with open(path) as f:
        for line in f:
            parts = line.split("#", 1)[0].split()
            if len(parts) >= 2:
                c, hw = float(parts[0]), float(parts[1])
                regions.append([c - hw, c + hw])
    return np.array(regions).reshape(-1, 2)


def mask_regions(spec, extra, default):
    """load_mask(spec, default) plus extra [[lam_min, lam_max], ...] from the config."""
    regions = load_mask(spec, default)
    if len(extra):
        regions = np.vstack([regions, np.array(extra, dtype=float).reshape(-1, 2)])
    return regions


def in_regions(lam, regions):
    """True for wavelengths inside any of the regions."""
    out = np.zeros(len(lam), dtype=bool)
    for lo, hi in regions:
        out |= (lam >= lo) & (lam <= hi)
    return out


def spline_knots(x, spacing, min_points):
    """
    Interior knots every `spacing` in x (sorted), dropping any that would leave fewer
    than min_points between knots, so a spline bridges gaps such as masked lines.
    """
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
