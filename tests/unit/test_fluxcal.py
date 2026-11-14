"""
Flux calibration on a simulated standard: G191-B2B's tabulated (CALSPEC) spectrum, seen
through a known sensitivity curve and the APO extinction, recovered by KARP.
"""
import numpy as np
import pytest

from karp import fluxcal
from karp.config import DEFAULTS

LAM = np.linspace(3800.0, 6600.0, 2000)  # A per pixel ~1.4
EXPTIME, AIRMASS, INST_FWHM = 60.0, 1.3, 9.8


def true_sensitivity(lam):
    """mag: 2.5 log10(e-/s/A per erg/s/cm2/A)"""
    x = (lam - 5200) / 1000
    return 38.0 - 0.8 * x ** 2 + 0.2 * x


def observe(ref, shift=0.0, noise=0.003, seed=6):
    """Electrons per pixel for the standard; `shift` (A) moves the observed wavelengths."""
    ref_lam, ref_flux, ref_fwhm = ref
    # The tabulated spectrum at KOSMOS's resolution (it is coarser than CALSPEC's here)
    fine = np.arange(LAM[0] - 50, LAM[-1] + 50, fluxcal.STEP)
    tab = np.interp(fine, ref_lam, ref_flux)
    tab = fluxcal.smooth_variable(tab, np.sqrt(INST_FWHM ** 2 - np.interp(fine, ref_lam, ref_fwhm) ** 2))
    f_true = np.interp(LAM - shift, fine, tab)
    rate = f_true * 10 ** (true_sensitivity(LAM) / 2.5)  # above the atmosphere, e-/s/A
    ext_lam, k = fluxcal.load_extinction("apo")
    rate = rate / 10 ** (0.4 * np.interp(LAM, ext_lam, k) * AIRMASS)
    electrons = rate * EXPTIME * fluxcal.pixel_width(LAM)
    sigma = noise * electrons
    rng = np.random.default_rng(seed)
    return electrons + rng.normal(0, 1, len(LAM)) * sigma, sigma, f_true


@pytest.fixture(scope="module")
def ref():
    lam, flux, fwhm, _ = fluxcal.load_standard("g191b2b")
    return lam, flux, fwhm


def run(ref, flux, sigma, **cfg):
    cfg = dict(DEFAULTS["fluxcal"], **cfg)
    ext = fluxcal.load_extinction("apo")
    return fluxcal.sensitivity(LAM, flux, sigma, np.zeros(len(LAM), bool), EXPTIME, AIRMASS, ext, ref,
                               INST_FWHM, cfg)


def test_sensitivity_and_closure(ref):
    flux, sigma, f_true = observe(ref)
    sens = run(ref, flux, sigma, fit_offset=False)
    lo, hi = sens["range"]
    inside = (LAM >= lo) & (LAM <= hi)
    assert lo < 3900 and hi > 6500
    assert np.max(np.abs(sens["fit"][inside] - true_sensitivity(LAM[inside]))) < 0.01  # mag
    # Closure: the standard calibrated with its own sensitivity gives its tabulated flux
    ext = fluxcal.load_extinction("apo")
    cal, _, valid, _ = fluxcal.calibrate(LAM, flux, sigma, EXPTIME, AIRMASS, ext, sens["lam"], sens["fit"],
                                         sens["range"])
    ratio = cal[valid & inside] / f_true[valid & inside]
    assert abs(np.median(ratio) - 1) < 0.003
    assert np.std(ratio) < 0.01


def test_wavelength_offset_found(ref):
    flux, sigma, _ = observe(ref, shift=1.5)
    sens = run(ref, flux, sigma, fit_offset=True)
    assert sens["offset"] == pytest.approx(1.5, abs=0.15)


def test_calibrate_extrapolates_and_stops():
    lam = np.linspace(4000, 6000, 201)
    sens_fit = np.full_like(lam, 30.0)
    ext = (np.array([3000.0, 11000.0]), np.zeros(2))
    flux, _, calibrated, extrapolated = fluxcal.calibrate(lam, np.ones_like(lam), np.ones_like(lam), 1.0, 1.0,
                                                          ext, lam, sens_fit, (4300.0, 5700.0), 100.0)
    assert np.all(calibrated[(lam >= 4200) & (lam <= 5800)])
    assert np.all(extrapolated[(lam >= 4200) & (lam < 4300)]) and not np.any(extrapolated[(lam > 4300) & (lam < 5700)])
    assert np.all(np.isnan(flux[lam < 4200])) and np.all(np.isnan(flux[lam > 5800]))


def test_standard_names_matched_loosely():
    assert fluxcal.find_standard("G191-B2B") == "g191b2b"
    assert fluxcal.find_standard("bd+28 4211") == "bd28d4211"
    assert fluxcal.find_standard("J0543") is None
