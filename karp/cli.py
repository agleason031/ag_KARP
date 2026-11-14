"""
KARP: KOSMOS Astronomical Reduction Pipeline. Command line and run control.

    python -m karp config.toml          # reduce every object found (or: karp, python ag_KARP.py)
    python -m karp config.toml --list   # only show which frames would be used

Created by Alex Gleason, based on original code by Chase L. Smith and Max Moe

><(((º>
"""
import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from datetime import timedelta

import numpy as np
from astropy import log as astropy_log

from . import __version__, calibration, fluxcal
from .config import ConfigError, load_config
from .discover import discover
from .pipeline import ObjectReduction, OutputPaths
from .plots import Plotter

# All KARP messages go through this logger: INFO and up to the console (DEBUG too with
# verbose = true), and everything to KARP_log.txt in each object's folder
log = logging.getLogger("karp")


def _log_warning(message, category, filename, lineno, file=None, line=None):
    """Python warnings during a run go to the log at DEBUG (KARP_log.txt; console only with verbose)."""
    log.debug(f"{category.__name__}: {message} ({filename}:{lineno})")


def play_finish_sound():
    """Play two short tones (777 Hz then 600 Hz) to signal KARP has finished."""
    import pygame # Imported here so pygame only touches the audio device when sound is on

    duration = 0.07 # seconds
    sample_rate = 44100
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    for freq in (777, 600): # Hz
        audio = np.int16(np.sin(2 * np.pi * freq * t) * 32767).tobytes()
        pygame.mixer.Sound(buffer=audio).play()
        pygame.time.delay(int(duration * 1000)) # wait for playback to finish


def setup_console_log(verbose):
    """Console output for the "karp" logger (added once, even if main() runs again)."""
    log.setLevel(logging.DEBUG)
    log.propagate = False
    console = next((h for h in log.handlers if getattr(h, "karp_console", False)), None)
    if console is None:
        console = logging.StreamHandler(sys.stdout)
        console.karp_console = True
        console.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(console)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)


def write_run_info(cfg, group, out):
    """Record the settings and frames used, for provenance."""
    with open(out.run_info, "w") as f:
        f.write(group.describe() + "\n\n")
        f.write("Settings:\n")
        f.write(json.dumps(cfg, indent=2) + "\n")


def standard_name(cfg, group):
    """Library name (or OBJNAME, with [fluxcal] reference) if this group is a flux standard, else None."""
    std = cfg["fluxcal"]["standard"]
    if std == "auto":
        return fluxcal.find_standard(group.objid)
    return std if group.objid == std else None


def run_group(cfg, group, sensfuncs=None):
    """
    Reduce one object/setup group and return its ObjectReduction. Everything logged
    meanwhile also goes to KARP_log.txt in the object's folder (rewritten each run).
    sensfuncs: {setup: [sensitivity functions]} shared between groups; a standard star
    adds its own, and any group with the same setup is flux calibrated with the one
    nearest in time (standards are run first, see main).
    """
    sensfuncs = {} if sensfuncs is None else sensfuncs
    steps = cfg["steps"]
    out = OutputPaths(cfg["paths"]["output_dir"], group.out_name, group.prefix or group.objid)
    out.make_dirs()
    write_run_info(cfg, group, out)

    log_file = logging.FileHandler(out.log, mode="w")
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(log_file)
    start = time.perf_counter()
    label = group.out_name if group.out_name == group.objid else f"{group.out_name} (OBJNAME {group.objid})"
    try:
        log.info(f"KARP reducing {label}")
        log.debug(group.describe())
        plots = Plotter(group.objid, out, cfg["output"]["verbose"])
        obj = ObjectReduction(cfg, group, out, plots)

        if steps["reduce"]:
            bias, flat = calibration.master_calibrations(group.bias, group.flat, cfg["detector"], cfg["flat"],
                                                         out, plots)
            for frame in sorted(group.science):
                obj.reduce_frame(frame, bias, flat)

        # Scale and combine the extracted spectra (read from disk, so this also works on
        # an earlier run's output)
        if steps["combine"]:
            obj.combine()

        # Optional: continuum-normalize the combined spectrum
        if steps["normalize"]:
            obj.normalize()

        # Optional: flux calibration, with a standard star observed with the same setup
        if steps["fluxcal"]:
            std = standard_name(cfg, group)
            if std is not None:
                sensfuncs.setdefault(group.setup, []).append(obj.sensitivity_function(std))
            available = sensfuncs.get(group.setup, [])
            if available:
                _, _, mjd = obj.reference_exposure()
                obj.flux_calibrate(min(available, key=lambda s: abs(s["mjd"] - mjd)))
            else:
                log.info("No standard star observed with this setup: flux calibration skipped")
        log.info(f"{group.out_name} took {timedelta(seconds=time.perf_counter() - start)}")
    except Exception as err:
        # One line on the console; the traceback goes to KARP_log.txt (and the console with verbose)
        log.error(f"KARP stopped with an error while reducing {label}: {type(err).__name__}: {err}")
        log.debug(traceback.format_exc())
        raise
    finally:
        log.removeHandler(log_file)
        log_file.close()
    return obj


def run(config, overrides=None, list_only=False):
    """
    Run KARP on a config file. `overrides` is an optional {"section.key": value} dict
    applied on top of the file. Each object is reduced in turn; an object that fails is
    logged and the others still run. Returns (results, failures): {output name:
    ObjectReduction} for the objects reduced, {output name: error} for those that failed.
    """
    try:
        cfg = load_config(config, overrides)
    except (OSError, ConfigError) as err:
        raise SystemExit(f"KARP could not read the configuration: {err}")

    setup_console_log(cfg["output"]["verbose"])
    with warnings.catch_warnings():
        # Library warnings (FITS header fixes, all-NaN slices, ...) are routine here;
        # keep them out of the console but in the log. Restored when the run ends.
        warnings.simplefilter("default")
        warnings.showwarning = _log_warning
        astropy_level = astropy_log.level
        astropy_log.setLevel("ERROR")
        try:
            return _run(cfg, list_only)
        finally:
            astropy_log.setLevel(astropy_level)


def _run(cfg, list_only):
    try:
        groups, problems = discover(cfg)
    except (OSError, ValueError) as err:
        raise SystemExit(f"KARP could not sort the raw frames: {err}")
    log.info(f"KARP found {len(groups)} object(s) to reduce in {cfg['paths']['raw_dir']}:")
    for g in groups:
        log.info(g.describe())
    for p in problems:
        log.warning(f"Skipping: {p}")
    target = cfg["target"]
    objects = sorted({g.objid for g in groups})
    if (target["ra"] is not None or target["dec"] is not None) and len(objects) > 1:
        log.warning(f"WARNING: [target] ra/dec are set and are used for every object ({', '.join(objects)}), "
                    "so the airmass and heliocentric correction will be wrong for all but one of them. "
                    "Remove them to use each frame's header, or limit the run with [frames] objects.")
    if list_only or not groups:
        return {}, {}

    log.info("KARP is swimming! (Code is running)")
    for i in range(6):
        log.info("- " * i + "><(((º>")
    start = time.perf_counter()

    # Standards first, so their sensitivity functions are ready for the other objects
    groups = sorted(groups, key=lambda g: standard_name(cfg, g) is None)
    results, failures, sensfuncs = {}, {}, {}
    for group in groups:
        log.info(f"\n=== {group.out_name} ===")
        try:
            results[group.out_name] = run_group(cfg, group, sensfuncs)
        except Exception as err: # logged by run_group; carry on with the next object
            failures[group.out_name] = f"{type(err).__name__}: {err}"

    if cfg["output"]["sound"]:
        play_finish_sound()

    log.info(f"\nKARP took: {timedelta(seconds=time.perf_counter() - start)} to run")
    log.info(f"Reduced {len(results)} of {len(groups)} object(s)" + (f": {', '.join(results)}" if results else ""))
    if failures:
        log.error(f"{len(failures)} object(s) failed (details in each one's KARP_log.txt):")
        for name, err in failures.items():
            log.error(f"  {name}: {err}")
    log.info("><(((º>")
    return results, failures


def main(argv=None):
    """Command line: python -m karp config.toml [--list]. Returns the exit status (1 if any object failed)."""
    parser = argparse.ArgumentParser(prog="karp", description="KOSMOS Astronomical Reduction Pipeline")
    parser.add_argument("config", help="TOML configuration file (see example_config.toml)")
    parser.add_argument("--list", action="store_true", help="show the frames KARP would use, then stop")
    parser.add_argument("--version", action="version", version=f"KARP {__version__}")
    args = parser.parse_args(argv)
    _, failures = run(args.config, list_only=args.list)
    return 1 if failures else 0
