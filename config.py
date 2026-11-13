"""
KARP configuration: load a TOML file, fill in defaults, and check it.

Every setting has a default here except paths.raw_dir and paths.output_dir.
Unknown sections or keys are errors, so typos are caught instead of ignored.
See example_config.toml for a documented example.
"""
import copy
import tomllib

class _Required:
    """Marker for settings with no default (survives deepcopy, so `is REQUIRED` works)."""
    def __deepcopy__(self, memo):
        return self
    def __repr__(self):
        return "REQUIRED"

REQUIRED = _Required()

# Section -> {key: default}. None means "not set" (TOML has no null; leave the key out).
DEFAULTS = {
    "paths": {
        "raw_dir": REQUIRED, # Directory with the raw KOSMOS frames
        "output_dir": REQUIRED, # Each object is reduced into output_dir/<object>/
        "pattern": "*.fits", # Which files in raw_dir to consider
    },
    "frames": {
        # Frame discovery is automatic (from FITS headers). These narrow or override it.
        # Frame numbers are the digits before ".fits" in the file name (k.0061.fits -> 61).
        "objects": [], # Only reduce these OBJNAMEs (empty = all objects found)
        "science": [], # Only use these science frames (empty = all)
        "bias": [], # Use exactly these bias frames (empty = all biases with matching binning)
        "flat": [], # Use exactly these flats (empty = all flats with matching setup)
        "arc": None, # Use this arc frame (default = matching argon arc nearest in time)
        "exclude": [], # Never use these frames
    },
    "target": {
        # Coordinates come from each frame's RA/DEC header; set these to override.
        "ra": None, # e.g. "23:50:59.00"
        "dec": None, # e.g. "+2:36:15.5"
        "site_height": 2788.0, # Observatory height in m (not in the headers; APO = 2788)
    },
    "steps": {
        "reduce": True, # Calibrate, trace, extract and wavelength-calibrate each frame
        "normalize": True, # Continuum-normalize each frame
        "combine": True, # Median-combine the normalized frames
        "fit_lines": False, # Velocity and metal-line fits on the combined spectrum
        "optimize": False, # Search aperture / normalization settings for maximum S/N
    },
    "detector": {
        # KOSMOS at 2x1 binning: two amplifiers split at column 512. Column ranges are
        # 0-based and end-exclusive; one entry per amplifier.
        "amp_data": [[0, 512], [512, 1024]],
        "amp_overscan": [[1034, 1074], [1084, 1124]], # First 10 overscan columns skipped
        "gain": [0.6, 0.6], # e-/DN (spec sheet; flats suggest ~0.72 / ~0.77)
        "readnoise": 5.7, # e- (spec sheet: 6)
        "full_well": 150000.0, # e-
        "overscan_order": 3, # Chebyshev order of the overscan fit along rows
    },
    "trace": {
        "order": 3, # Polynomial order of the trace fit
        "col": None, # Starting column for the trace search (default: brightest column)
    },
    "extraction": {
        "aperture": 20, # Full aperture width in pixels (an odd width w uses (w-1)/2 per side)
        "buffer": 20, # Gap between aperture and background region, pixels per side
        "background": 25, # Background region width, pixels per side
    },
    "cosmic_rays": {
        "enabled": True, # L.A.Cosmic on each 2D science frame
        "sigclip": 4.5, # Detection threshold in sigma
    },
    "normalization": {
        "line_width": 250, # Half-width (pixels) of the local line fit at each point
        "boxcar": 70, # Boxcar (pixels) used to smooth the fitted continuum
        "clip": None, # Mask normalized flux >= this (default: no masking)
    },
    "combine": {
        "clip": None, # Mask combined flux > this (default: no masking)
        "clip_regions": [], # [[lam_min, lam_max, threshold], ...]: mask flux above threshold there
        "jackknife": False, # Also report S/N with each frame left out
    },
    "lines": {
        "metal_mask": [2.0] * 16, # Half-widths (A) of the 16 metal-line windows for fit_lines
    },
    "output": {
        "verbose": False, # Extra diagnostics, plots and intermediate files
        "sound": False, # Play a short tone when KARP finishes
    },
}


class ConfigError(ValueError):
    pass


def _check_type(section, key, value, default):
    """Values must match the type of their default (ints are accepted for floats)."""
    if default is None or default is REQUIRED:
        return value
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, type(default))
    if not ok:
        raise ConfigError(f"[{section}] {key} should be {type(default).__name__}, got {value!r}")
    return value


def merge(user):
    """Apply defaults to a user config dict and validate it."""
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in user.items():
        if section not in DEFAULTS:
            raise ConfigError(f"Unknown section [{section}]")
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] should be a table")
        for key, value in values.items():
            if key not in DEFAULTS[section]:
                raise ConfigError(f"Unknown setting [{section}] {key}")
            cfg[section][key] = _check_type(section, key, value, DEFAULTS[section][key])
    for section, values in cfg.items():
        for key, value in values.items():
            if value is REQUIRED:
                raise ConfigError(f"[{section}] {key} is required")

    det = cfg["detector"]
    if not (len(det["amp_data"]) == len(det["amp_overscan"]) == len(det["gain"])):
        raise ConfigError("[detector] amp_data, amp_overscan and gain need one entry per amplifier")
    if len(cfg["lines"]["metal_mask"]) != 16:
        raise ConfigError("[lines] metal_mask needs 16 values (one per metal line)")
    return cfg


def load_config(path, overrides=None):
    """
    Read a TOML config file and return the full settings dict.
    `overrides` is an optional {"section.key": value} dict applied on top of the file.
    """
    with open(path, "rb") as f:
        user = tomllib.load(f)
    for dotted, value in (overrides or {}).items():
        section, key = dotted.split(".", 1)
        user.setdefault(section, {})[key] = value
    return merge(user)
