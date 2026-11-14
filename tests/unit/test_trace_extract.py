"""Tracing and extraction on a simulated 2D frame: a tilted star of known brightness on sky."""
import numpy as np
import pytest

from karp.extract import extract, variance_image
from karp.trace import _fit_row, find_slit_sources, fit_trace, nearest_source

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


def test_row_without_a_peak_is_skipped():
    """A row fainter in the aperture than in the background windows gives NaN, not an error."""
    x = np.arange(10, 20)
    assert np.isnan(_fit_row(x, np.full(10, 50.0), np.full(20, 100.0), 15)).all()  # amplitude guess < 0
    assert np.isnan(_fit_row(x, np.full(10, -300.0), np.full(20, -300.0), 15)).all()  # no peak, background < -100
    y = 1000 * np.exp(-(x - 15.0) ** 2 / 8) - 300.0  # a real peak on a very negative background
    assert _fit_row(x, y, np.full(20, -300.0), 15)[1] == pytest.approx(15.0, abs=0.01)


@pytest.fixture(scope="module")
def two_stars():
    """A faint target near the slit center and a 10x brighter star far from it."""
    rng = np.random.default_rng(2)
    n_rows, n_cols = 600, 400
    cols = np.arange(n_cols)
    model = np.full((n_rows, n_cols), SKY)
    for col, flux in ((190.0, 3000.0), (330.0, 30000.0)):
        prof = np.exp(-(cols - col) ** 2 / (2 * PSF_SIGMA ** 2))
        model += flux * prof / prof.sum()
    return rng.poisson(model).astype(float) + rng.normal(0, READNOISE, model.shape)


def test_slit_sources_and_nearest_to_center(two_stars):
    _, sources = find_slit_sources(two_stars, (2, 2))
    assert [c for c, _ in sources] == [190, 330]
    assert sources[1][1] > 5 * sources[0][1]
    assert nearest_source(sources, 200)[0] == 190
    assert nearest_source([], 200) is None


def test_trace_follows_the_chosen_star(two_stars):
    args = (two_stars, APERTURE_HALF, BUFFER, BACKGROUND, 1, (2, 2))
    _, brightest, _, _ = fit_trace(*args)
    _, chosen, _, _ = fit_trace(*args, start_col=192)
    assert np.allclose(brightest, 330, atol=0.2)
    assert np.allclose(chosen, 190, atol=0.2)
