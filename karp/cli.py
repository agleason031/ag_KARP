"""
KARP: KOSMOS Astronomical Reduction Pipeline. Command line and run control.

    python -m karp config.toml          # reduce every object found (or: karp, python ag_KARP.py)
    python -m karp config.toml --list   # only show which frames would be used

Originally by Chase L. Smith and Max Moe; reworked by Alex Gleason.

><(((º>
"""
import argparse
import json
import logging
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
from astropy import log as astropy_log

from . import calibration, fluxcal
from .config import ConfigError, load_config
from .discover import discover
from .pipeline import ObjectReduction, OutputPaths
from .plots import Plotter

astropy_log.setLevel("ERROR")
warnings.filterwarnings("ignore")

# All KARP messages go through this logger: INFO and up to the console (DEBUG too with
# verbose = true), and everything to KARP_log.txt in each object's folder
log = logging.getLogger("karp")


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
    out = OutputPaths(cfg["paths"]["output_dir"], group.out_name, group.objid)
    out.make_dirs()
    write_run_info(cfg, group, out)

    log_file = logging.FileHandler(out.log, mode="w")
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(log_file)
    start = time.perf_counter()
    try:
        log.info(f"KARP reducing {group.objid}")
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
        log.info(f"{group.objid} took {timedelta(seconds=time.perf_counter() - start)}")
    except Exception:
        log.exception(f"KARP stopped with an error while reducing {group.objid}")
        raise
    finally:
        log.removeHandler(log_file)
        log_file.close()
    return obj


def main(argv=None, overrides=None):
    """
    Run KARP. `argv` are command-line arguments (default: sys.argv); `overrides` is an
    optional {"section.key": value} dict applied on top of the config file.
    Returns {output name: ObjectReduction} for the groups that were reduced.
    """
    parser = argparse.ArgumentParser(prog="karp", description="KOSMOS Astronomical Reduction Pipeline")
    parser.add_argument("config", help="TOML configuration file (see example_config.toml)")
    parser.add_argument("--list", action="store_true", help="show the frames KARP would use, then stop")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides)
    except (OSError, ConfigError) as err:
        raise SystemExit(f"KARP could not read the configuration: {err}")

    setup_console_log(cfg["output"]["verbose"])

    groups, problems = discover(cfg)
    log.info(f"KARP found {len(groups)} object(s) to reduce in {cfg['paths']['raw_dir']}:")
    for g in groups:
        log.info(g.describe())
    for p in problems:
        log.warning(f"Skipping: {p}")
    if args.list or not groups:
        return {}

    log.info("KARP is swimming! (Code is running)")
    for i in range(6):
        log.info("- " * i + "><(((º>")
    start = time.perf_counter()

    # Standards first, so their sensitivity functions are ready for the other objects
    groups = sorted(groups, key=lambda g: standard_name(cfg, g) is None)
    results, sensfuncs = {}, {}
    for group in groups:
        log.info(f"\n=== {group.objid} ===")
        results[group.out_name] = run_group(cfg, group, sensfuncs)

    if cfg["output"]["sound"]:
        play_finish_sound()

    log.info(f"KARP took: {timedelta(seconds=time.perf_counter() - start)} to run")
    log.info("><(((º>")
    return results
