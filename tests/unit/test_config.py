"""Settings: defaults, type checks and the errors for bad values."""
import pytest

from karp.config import DEFAULTS, ConfigError, load_config, merge

PATHS = {"paths": {"raw_dir": "raw", "output_dir": "out"}}


def test_defaults_filled_in():
    cfg = merge(PATHS)
    assert cfg["paths"]["layout"] == "night"
    assert cfg["combine"]["method"] == "weighted"
    assert cfg["extraction"]["aperture_arcsec"] == DEFAULTS["extraction"]["aperture_arcsec"]
    assert cfg["target"]["ra"] is None


def test_int_accepted_for_float():
    cfg = merge({**PATHS, "extraction": {"aperture_arcsec": 8}})
    assert cfg["extraction"]["aperture_arcsec"] == 8.0
    assert isinstance(cfg["extraction"]["aperture_arcsec"], float)


@pytest.mark.parametrize("user, message", [
    ({}, "raw_dir is required"),
    ({**PATHS, "extraktion": {}}, "Unknown section"),
    ({**PATHS, "combine": {"methd": "mean"}}, "Unknown setting"),
    ({**PATHS, "extraction": {"aperture": 20}}, "aperture_arcsec"),  # old pixel setting: hint
    ({**PATHS, "output": {"verbose": "yes"}}, "should be bool"),
    ({**PATHS, "trace": {"order": 2.5}}, "should be int"),
    ({**PATHS, "combine": {"method": "average"}}, "method"),
    ({**PATHS, "combine": {"reject_sigma": -1}}, "reject_sigma"),
    ({"paths": {**PATHS["paths"], "layout": "nights"}}, "layout"),
    ({**PATHS, "wavecal": {"order": 0}}, "order"),
    ({**PATHS, "wavecal": {"flexure": "fix"}}, "flexure"),
    ({**PATHS, "flat": {"zeroth_order": [[100]]}}, "zeroth_order"),
    ({**PATHS, "fluxcal": {"reference": "star.txt"}}, "reference needs standard"),
    ({**PATHS, "detector": {"amp_data": [[0, 512], [512, 1024]]}}, "together"),
    ({**PATHS, "extraction": {"aperture_arcsec": 0.0}}, "must be > 0"),
])
def test_bad_settings_rejected(user, message):
    with pytest.raises(ConfigError, match=message):
        merge(user)


def test_example_config_is_valid():
    cfg = load_config("example_config.toml")
    assert cfg["paths"]["layout"] == "objects"


def test_overrides_applied(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[paths]\nraw_dir = "a"\noutput_dir = "b"\n')
    cfg = load_config(path, {"combine.method": "median", "paths.layout": "objects"})
    assert cfg["combine"]["method"] == "median" and cfg["paths"]["layout"] == "objects"
