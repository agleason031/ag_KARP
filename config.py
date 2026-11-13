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
        "pattern": "*.fits", # Which files in raw_dir to consider; "**/*.fits" includes subdirectories
    },
    "frames": {
        # Frame discovery is automatic (from FITS headers). These narrow or override it.
        # Frame numbers are the digits before ".fits" in the file name (k.0061.fits -> 61).
        "objects": [], # Only reduce these OBJNAMEs (empty = all objects found)
        "science": [], # Only use these science frames (empty = all)
        "bias": [], # Use exactly these bias frames (empty = all biases with matching binning)
        "flat": [], # Use exactly these flats (empty = all flats with matching setup; none found =
                    # no flat-field correction)
        "arc": [], # Use exactly these arcs (empty = for each lamp combination, the matching arcs
                   # nearest in time); they are median-combined per lamp combination, then added
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
        "combine": True, # Scale and combine the extracted frames
        "normalize": True, # Continuum-normalize the combined spectrum (optional)
        "fluxcal": True, # Flux-calibrate with a standard star of the same setup (skipped if none)
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
    "flat": {
        # The zeroth order of the grating crosses the slit as a bright band in the flat-lamp
        # exposures; the final flat is set to 1 in those rows (over the illuminated columns).
        "zeroth_order": "auto", # "auto" (find bright bands in the flat), "none", or
                                # [[row_min, row_max], ...] (inclusive, trimmed-frame rows)
        "zeroth_threshold": 0.02, # "auto": rows this much (fraction) above the running median
        "zeroth_pad": 10, # "auto": rows added on each side of a band
    },
    "trace": {
        "order": 3, # Polynomial order of the trace fit
        "col": None, # Starting column for the trace search (default: brightest column)
    },
    "extraction": {
        "aperture": 20, # Full aperture width in pixels (an odd width w uses (w-1)/2 per side)
        "buffer": 20, # Gap between aperture and background region, pixels per side
        "background": 25, # Background region width, pixels per side
        "method": "optimal", # "optimal" (Horne 1986) or "boxcar"
        "sky_order": 1, # Polynomial order of the sky fit across the slit (per row)
        "profile_window": 101, # Rows over which the spatial profile is smoothed (optimal)
        "reject_sigma": 5.0, # Optimal extraction: leave out pixels deviating by more than this
    },
    "cosmic_rays": {
        "enabled": True, # L.A.Cosmic on each 2D science frame
        "sigclip": 4.5, # Detection threshold in sigma
    },
    "normalization": {
        # Optional: fit a continuum to the combined spectrum and divide by it
        # (a cubic spline with asymmetric sigma clipping; see normalize.py).
        "knot_spacing": 150.0, # A between spline knots; must exceed the widest masked region
        "low_reject": 2.0, # Leave out pixels more than this many sigma below the continuum
        "high_reject": 3.0, # ... or this many sigma above it
        "mask": "default", # Regions left out of the fit: "default" (calib_data/masks/stellar.txt),
                           # "none", or a file with columns center, half-width (A)
        "extra_mask": [], # More regions to leave out: [[lam_min, lam_max], ...]
        "clip": None, # Flag (Mask bit 1) normalized flux above this (default: no flag)
        "clip_regions": [], # [[lam_min, lam_max, threshold], ...]: flag normalized flux above threshold there
    },
    "fluxcal": {
        # Optional: a spectrophotometric standard observed with the same setup (disperser,
        # slit, binning) gives the sensitivity function for every object with that setup.
        "standard": "auto", # "auto": objects whose OBJNAME matches calib_data/standards/index.json;
                            # or the OBJNAME of the one object to use as the standard
        "reference": None, # Tabulated flux of the standard (air wavelength A, erg/s/cm2/A) if it
                           # isn't in the library (use with standard = its OBJNAME)
        "extinction": "apo", # Extinction curve: "apo" or a file of wavelength (A), mag/airmass
        "knot_spacing": 250.0, # A between spline knots of the sensitivity fit
        "clip_sigma": 3.0, # Sigma clipping of the sensitivity fit
        "mask": "default", # Regions left out of the fit: "default" (calib_data/masks/standard.txt),
                           # "none", or a file with columns center, half-width (A)
        "extra_mask": [], # More regions to leave out: [[lam_min, lam_max], ...]
        "resolution": 0.0, # KOSMOS resolution (FWHM, A) for comparing the standard with its
                           # tabulated flux; 0 = the arc line width (the result depends little on it)
        "fit_offset": True, # Fit and take out a wavelength offset between the standard and its
                            # tabulated spectrum before comparing them (logged as a wavelength check)
        "max_offset": 5.0, # A, largest offset searched
        "max_extrapolate": 100.0, # A the sensitivity is extended (linearly, Mask bit 8) past the
                                  # fitted range, e.g. when a masked line sits at the detector edge
    },
    "combine": {
        "scale": "smooth", # Scale frames to the best one: "smooth" (polynomial in wavelength),
                           # "median" (one factor), or "none"
        "scale_order": 2, # Polynomial order for scale = "smooth"
        "method": "weighted", # "weighted" (inverse-variance mean with outlier rejection),
                              # "mean" (plain mean) or "median"
        "reject_sigma": 5.0, # method = "weighted": reject values this many sigma (their own
                             # error) from the median of the frames; 0 = no rejection
        "jackknife": False, # Also report S/N with each frame left out
    },
    "wavecal": {
        "order": "auto", # Polynomial order, or "auto": the order (2-6) that best predicts each
                         # line from the others (leave-one-out)
        "lamps": [], # Line lists to use, e.g. ["ARGON"] (default: lamps that are on in the arc header)
        "reference": None, # Reference solution file (default: calib_data/wavecal/kosmos_<disperser>.json)
        "search": 200, # Largest shift (rows) allowed between the reference and this arc
        "match_tol": 3.0, # Match observed and catalog lines within this many rows
        "clip_sigma": 3.0, # Sigma clipping of the fit
        "min_lines": 8, # Fewest matched lines for a solution
        "max_rms": 15.0, # km/s; a worse solution stops with an error (likely misidentified lines)
        "peak_sigma": 5.0, # Arc line detection threshold (times the noise)
        "flexure": "report", # Night-sky line check: "off", "report", or "correct" (shift wavelengths)
        "sky_lines": [5577.338, 6300.304, 6363.776], # Night-sky lines for the flexure check (air, A)
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
    if default is None or default is REQUIRED or (section, key) in (("wavecal", "order"), ("flat", "zeroth_order")):
        return value # checked in merge()
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
    if isinstance(user.get("frames", {}).get("arc"), int): # a single arc frame number
        user = {**user, "frames": {**user["frames"], "arc": [user["frames"]["arc"]]}}
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

    if cfg["combine"]["scale"] not in ("smooth", "median", "none"):
        raise ConfigError('[combine] scale should be "smooth", "median" or "none"')
    if cfg["combine"]["method"] not in ("weighted", "mean", "median"):
        raise ConfigError('[combine] method should be "weighted", "mean" or "median"')
    if not isinstance(cfg["combine"]["reject_sigma"], (int, float)) or cfg["combine"]["reject_sigma"] < 0:
        raise ConfigError("[combine] reject_sigma should be a number >= 0 (0 = no rejection)")
    if cfg["extraction"]["method"] not in ("optimal", "boxcar"):
        raise ConfigError('[extraction] method should be "optimal" or "boxcar"')

    wc = cfg["wavecal"]
    if wc["order"] != "auto" and not (isinstance(wc["order"], int) and wc["order"] >= 1):
        raise ConfigError('[wavecal] order should be "auto" or a positive integer')
    if wc["flexure"] not in ("off", "report", "correct"):
        raise ConfigError('[wavecal] flexure should be "off", "report" or "correct"')

    if cfg["fluxcal"]["standard"] != "auto" and not isinstance(cfg["fluxcal"]["standard"], str):
        raise ConfigError('[fluxcal] standard should be "auto" or an object name')
    if cfg["fluxcal"]["reference"] is not None and cfg["fluxcal"]["standard"] == "auto":
        raise ConfigError("[fluxcal] reference needs standard = the OBJNAME of the standard star")

    zo = cfg["flat"]["zeroth_order"]
    if not (zo in ("auto", "none") or (isinstance(zo, list) and all(
            isinstance(r, list) and len(r) == 2 and all(isinstance(v, int) for v in r) for r in zo))):
        raise ConfigError('[flat] zeroth_order should be "auto", "none" or [[row_min, row_max], ...]')

    det = cfg["detector"]
    if not (len(det["amp_data"]) == len(det["amp_overscan"]) == len(det["gain"])):
        raise ConfigError("[detector] amp_data, amp_overscan and gain need one entry per amplifier")
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
