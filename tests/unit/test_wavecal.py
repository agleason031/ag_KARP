"""Arc line finding and the wavelength solution on simulated arcs with a known answer."""
import numpy as np
import pytest

from karp import wavecal
from karp.config import DEFAULTS

SIGMA = 1.5  # rows


def gauss(x, a, mu, s=SIGMA):
    return a * np.exp(-(x - mu) ** 2 / (2 * s ** 2))


def test_line_finder_centroids_and_cuts():
    x = np.arange(800, dtype=float)
    rng = np.random.default_rng(2)
    normal = [100.3, 150.7, 200.1, 250.5, 350.2, 400.9, 450.4, 600.6, 650.8, 700.0]
    spec = sum(gauss(x, 2000, mu) for mu in normal)
    spec += gauss(x, 20000, 300.0) + gauss(x, 3000, 306.5)  # faint line on a bright line's wing
    spec += gauss(x, 1500, 520.0, s=5.0)  # broad ghost (3x the line width)
    spec += rng.normal(0, 5, len(x))
    rows, amps, sigs, info = wavecal.find_arc_lines(spec)
    assert info["too_wide"] >= 1 and info["crowded"] >= 1
    assert not np.any(np.abs(rows - 520) < 5)  # ghost left out
    assert not np.any(np.abs(rows - 306.5) < 1)  # line on the wing left out
    for mu in normal + [300.0]:
        assert np.min(np.abs(rows - mu)) < 0.05
    assert info["fwhm"] == pytest.approx(2.3548 * SIGMA, rel=0.05)


@pytest.mark.parametrize("binning_y, shift", [(1, 0.0), (2, 3.4)])
def test_solution_recovers_known_dispersion(binning_y, shift):
    """
    An argon arc made from the real line list, placed by the reference solution moved by
    `shift` rows (as from flexure) and converted to this binning; a broad ghost is added.
    """
    ref = wavecal.load_reference("blue")
    n_rows = 4096 // binning_y
    truth_poly = wavecal.reference_polynomial(ref, binning_y)
    truth = lambda r: truth_poly(np.asarray(r, dtype=float) + shift)
    lo, hi = sorted([truth(0), truth(n_rows - 1)])
    catalog = wavecal.load_linelist(["ARGON"])
    lines = wavecal.remove_blends(catalog[(catalog > lo + 10) & (catalog < hi - 10)], 6.0)

    grid = np.linspace(-50, n_rows + 50, 20 * n_rows)
    lam_grid = truth(grid)
    order = np.argsort(lam_grid)
    line_rows = np.interp(lines, lam_grid[order], grid[order])
    rng = np.random.default_rng(3)
    x = np.arange(n_rows, dtype=float)
    spec = np.zeros(n_rows)
    for r in line_rows:
        spec += gauss(x, 10 ** rng.uniform(2.5, 4.5), r, SIGMA / binning_y ** 0.5)
    ghost_row = n_rows * 0.37
    spec += gauss(x, 3000, ghost_row, 10.0 / binning_y)
    spec += rng.normal(0, 3, n_rows)

    wc = DEFAULTS["wavecal"]
    wcfg = dict(wc, search=wc["search"] / binning_y, match_tol=wc["match_tol"] / binning_y)
    sol = wavecal.solve(spec, ["ARGON"], ref, binning_y, wcfg)

    assert sol.used.sum() >= 20
    assert sol.rms_kms < 1.0
    error = sol.poly(x) - truth(x)
    assert np.max(np.abs(error)) < 0.05  # A, over the whole detector
    assert not np.any(np.abs(sol.rows[sol.used] - ghost_row) < 3)


def test_unknown_disperser_message():
    with pytest.raises(FileNotFoundError, match="make_wavecal_reference"):
        wavecal.load_reference("purple")
