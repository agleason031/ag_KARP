"""Tracing and extraction on a simulated 2D frame: a tilted star of known brightness on sky."""
import numpy as np
import pytest

from karp.extract import extract, variance_image
from karp.trace import fit_trace

N_ROWS, N_COLS = 600, 160
STAR, SKY, READNOISE, PSF_SIGMA = 20000.0, 300.0, 5.7, 2.0
APERTURE_HALF, BUFFER, BACKGROUND = 9, 12, 20
CR_ROWS = [100, 250, 400]


def true_center(rows):
    """Tilted, slightly curved trace."""
    x = (rows - N_ROWS / 2) / N_ROWS
    return 80.0 + 6.0 * x + 3.0 * x ** 2


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(1)
    rows = np.arange(N_ROWS)
    cols = np.arange(N_COLS)
    center = true_center(rows)
    # Star: Gaussian profile integrated over each pixel would be exact; at sigma = 2 px
    # sampling at the pixel center loses < 1%, so normalize the sampled profile instead
    prof = np.exp(-(cols[None, :] - center[:, None]) ** 2 / (2 * PSF_SIGMA ** 2))
    prof /= prof.sum(axis=1, keepdims=True)
    model = STAR * prof + SKY
    image = rng.poisson(model).astype(float) + rng.normal(0, READNOISE, model.shape)
    clean = image.copy()
    for r in CR_ROWS:  # cosmic rays inside the aperture
        image[r, int(round(center[r])) + 2] += 30000.0
    return image, clean, center


def test_trace_follows_the_star(frame):
    image, _, center = frame
    _, trace, _, used = fit_trace(image, APERTURE_HALF, BUFFER, BACKGROUND, 3, (2, 2))
    assert used.sum() > 0.9 * N_ROWS
    assert np.sqrt(np.mean((trace - center) ** 2)) < 0.05


@pytest.mark.parametrize("method", ["optimal", "boxcar"])
def test_extraction_recovers_flux_and_errors(frame, method):
    _, clean, center = frame
    var = variance_image(clean, np.ones_like(clean), READNOISE)
    res = extract(clean, var, np.ones_like(clean), center, APERTURE_HALF, BUFFER, BACKGROUND, READNOISE,
                  method=method, sky_order=1, profile_window=51)
    flux, sigma = res["flux"], np.sqrt(res["var"])
    assert np.isfinite(flux).all()
    assert abs(np.mean(flux) / STAR - 1) < 0.003
    assert np.median(res["sky"]) == pytest.approx(SKY * (2 * APERTURE_HALF + 1), rel=0.01)
    pull = (flux - STAR) / sigma  # errors should describe the actual scatter
    assert 0.85 < np.std(pull) < 1.15


def test_optimal_extraction_rejects_cosmic_rays(frame):
    image, _, center = frame
    var = variance_image(image, np.ones_like(image), READNOISE)
    args = (image, var, np.ones_like(image), center, APERTURE_HALF, BUFFER, BACKGROUND, READNOISE)
    opt = extract(*args, method="optimal", profile_window=51)
    box = extract(*args, method="boxcar")
    for r in CR_ROWS:
        assert opt["n_rejected"][r] >= 1
        assert abs(opt["flux"][r] - STAR) < 5 * np.sqrt(opt["var"][r])
        assert box["flux"][r] - STAR > 20000  # boxcar keeps the cosmic ray
