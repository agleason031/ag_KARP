"""Combining frames and continuum normalization on simulated spectra with a known answer."""
import numpy as np
import pytest

from karp.combine import combine_frames
from karp.normalize import fit_continuum
from karp.regions import mask_regions

LAM = np.linspace(4000.0, 6500.0, 2500)


def star(lam):
    """A smooth continuum with absorption lines (electrons per pixel)."""
    x = (lam - 5000) / 1000
    cont = 20000 * (1 + 0.3 * x - 0.2 * x ** 2)
    lines = 1 - 0.5 * np.exp(-(lam - 4861.3) ** 2 / (2 * 8.0 ** 2)) - 0.4 * np.exp(-(lam - 4340.5) ** 2 / (2 * 7.0 ** 2))
    for center in (4500.0, 5170.0, 5900.0, 6100.0):
        lines -= 0.3 * np.exp(-(lam - center) ** 2 / (2 * 1.0 ** 2))
    return cont, cont * lines


def frames(spike=True):
    """Four noisy frames of the same spectrum, with different (smooth) scalings and a spike in one."""
    rng = np.random.default_rng(4)
    _, flux = star(LAM)
    x = (LAM - 5000) / 1000
    scales = {1: 1.0, 2: 0.8, 3: 0.9 * (1 + 0.05 * x), 4: 0.7}
    out = {}
    for n, scale in scales.items():
        f = flux * scale
        sigma = np.sqrt(f + 100.0)
        noisy = f + rng.normal(0, 1, len(LAM)) * sigma
        if spike and n == 3:
            noisy[1200] += 50 * sigma[1200]
        out[n] = (LAM.copy(), noisy, sigma)
    return out, flux


def test_weighted_combine_rejects_outlier_and_gives_right_errors():
    data, truth = frames()
    comb = combine_frames(data, "smooth", 2, "weighted", 5.0)
    assert comb.ref_frame == 1  # best S/N: frame 1 (scale 1), so the result is on the truth's scale
    pull = (comb.flux - truth) / comb.sigma
    assert abs(np.mean(pull)) < 0.1
    assert 0.9 < np.std(pull) < 1.1
    assert comb.mask[1200] & 4 and comb.n_rejected[3] >= 1
    assert abs(pull[1200]) < 5
    assert not np.any(comb.mask & 2)


@pytest.mark.parametrize("method", ["mean", "median"])
def test_other_methods(method):
    data, truth = frames(spike=False)
    comb = combine_frames(data, "smooth", 2, method, 5.0)
    pull = (comb.flux - truth) / comb.sigma
    assert abs(np.mean(pull)) < 0.1
    assert 0.8 < np.std(pull) < 1.2


def test_no_data_flagged():
    data, _ = frames(spike=False)
    lam, f, s = data[2]
    f = f.copy()
    f[:100] = np.nan
    data[2] = (lam, f, s)
    data = {2: data[2]}
    comb = combine_frames(data, "none", 2, "weighted", 5.0)
    assert np.all(comb.mask[:100] & 2) and not np.any(comb.mask[100:] & 2)


def test_continuum_recovered():
    rng = np.random.default_rng(5)
    cont, flux = star(LAM)
    sigma = 0.005 * flux
    noisy = flux + rng.normal(0, 1, len(LAM)) * sigma
    regions = mask_regions("default", [], "stellar.txt")
    fit, used, _ = fit_continuum(LAM, noisy, sigma, np.zeros(len(LAM), bool), regions)
    inner = (LAM > 4100) & (LAM < 6400)  # the ends are less constrained
    assert np.max(np.abs(fit[inner] / cont[inner] - 1)) < 0.01
    assert np.median(np.abs(fit / cont - 1)) < 0.002
    # the narrow unmasked lines are clipped out of the fit
    for center in (4500.0, 5170.0, 5900.0, 6100.0):
        assert not used[np.argmin(np.abs(LAM - center))]
