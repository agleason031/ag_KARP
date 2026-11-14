"""Detector layout from headers (no pixel data needed)."""
import pytest
from astropy.io import fits

from karp import detector
from karp.config import DEFAULTS

DET = dict(DEFAULTS["detector"])


def header(binx, biny, sections=None, overscan=50):
    """Header of a raw KOSMOS frame; sections: {amp: (DSEC, BSEC)} strings, or None."""
    h = fits.Header()
    h["NAXIS1"] = 2048 // binx + 2 * overscan
    h["NAXIS2"] = 4096 // biny
    h["BINX"], h["BINY"] = binx, biny
    for amp, (dsec, bsec) in (sections or {}).items():
        h[f"DSEC{amp}"], h[f"BSEC{amp}"] = dsec, bsec
    return h


@pytest.mark.parametrize("binx, biny", [(1, 1), (2, 1), (2, 2), (4, 4)])
def test_layout_from_binning(binx, biny):
    geom = detector.frame_geometry(header(binx, biny), DET)
    n = 1024 // binx
    assert geom["amp_data"] == [[0, n], [n, 2 * n]]
    skip = DET["overscan_skip"]
    assert geom["amp_overscan"] == [[2 * n + skip, 2 * n + 50], [2 * n + 50 + skip, 2 * n + 100]]
    assert geom["rows"] == [0, 4096 // biny]
    assert geom["source"].startswith("computed")


def test_layout_from_header_sections():
    # 2x2 frame whose DSEC/BSEC fit the frame: used as given (1-based, inclusive -> 0-based)
    h = header(2, 2, {1: ("[1:512,1:2048]", "[1025:1074,1:2048]"),
                      2: ("[513:1024,1:2048]", "[1075:1124,1:2048]")})
    geom = detector.frame_geometry(h, DET)
    assert geom["source"] == "header DSEC/BSEC"
    assert geom["amp_data"] == [[0, 512], [512, 1024]]
    assert geom["amp_overscan"] == [[1024 + 10, 1074], [1074 + 10, 1124]]


def test_unbinned_header_sections_ignored():
    # 2x1 frame carrying the unbinned sections (past the frame edge): computed from binning
    h = header(2, 1, {1: ("[1:1024,1:4096]", "[2049:2098,1:4096]"),
                      2: ("[1025:2048,1:4096]", "[2099:2148,1:4096]")})
    geom = detector.frame_geometry(h, DET)
    assert geom["source"].startswith("computed")
    assert geom["amp_data"] == [[0, 512], [512, 1024]]


def test_wrong_size_rejected():
    h = header(2, 2)
    h["NAXIS1"] = 1000
    with pytest.raises(detector.GeometryError):
        detector.frame_geometry(h, DET)


def test_config_layout_overrides_header():
    det = dict(DET, amp_data=[[0, 500], [500, 1000]], amp_overscan=[[1000, 1060], [1060, 1120]])
    geom = detector.frame_geometry(header(2, 2), det)
    assert geom["amp_data"] == [[0, 500], [500, 1000]] and geom["source"] == "[detector] config"


def test_parse_section():
    assert detector.parse_section("[1:512,1:2048]") == (0, 512, 0, 2048)
    assert detector.parse_section("[512:1,2048:1]") == (0, 512, 0, 2048)  # reversed ranges


def test_to_binned():
    assert detector.to_binned(101, 2, odd=True) == 51
    assert detector.to_binned(100, 2, odd=True) == 51
    assert detector.to_binned(4, 4, minimum=2) == 2
