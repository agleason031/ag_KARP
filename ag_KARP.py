#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KARP

KOSMOS Astronomical Reduction Pipeline

Created by Chase L. Smith, Max Moe

Usage:
    python ag_KARP.py config.toml          # reduce every object found
    python ag_KARP.py config.toml --list   # only show which frames would be used

><(((º>
"""

# Importing relevant packages
from datetime import timedelta
from astropy import log
log.setLevel('ERROR')
from grapher import grapher
import numpy as np
import argparse
import json
import logging
import sys
import warnings
import time
import os
import sci_tools
import sci_modules
from config import load_config, ConfigError
from discover import discover

#-------------------------------
#constant definitions

# Suppress warnings
warnings.filterwarnings("ignore")

# All KARP messages go through this logger: INFO and up to the console (DEBUG too with
# verbose = true), and everything to OUT/KARP_log.txt for each object
log = logging.getLogger("karp")

#----------------------------------
#function definitions

def play_finish_sound():
    """Play two short tones (777 Hz then 600 Hz) to signal KARP has finished."""
    import pygame # Imported here so pygame only touches the audio device when sound is on

    duration = 0.07 # seconds
    sample_rate = 44100
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    for freq in (777, 600): # Hz
        # Generate sine wave, convert to 16-bit signed integers, and play it
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

def write_run_info(cfg, group, target_dir):
    """Record the settings and frames used, for provenance."""
    with open(target_dir + "OUT/KARP_run_info.txt", "w") as f:
        f.write(group.describe() + "\n\n")
        f.write("Settings:\n")
        f.write(json.dumps(cfg, indent=2) + "\n")

def run_group(cfg, group):
    """
    Reduce one object/setup group. Returns its sci_modules instance. Everything logged
    meanwhile also goes to <output_dir>/<object>/OUT/KARP_log.txt (rewritten each run).
    """
    steps = cfg["steps"]
    verbose = cfg["output"]["verbose"]
    target_dir = os.path.join(cfg["paths"]["output_dir"], group.out_name) + "/"
    os.makedirs(target_dir + "OUT", exist_ok=True) # Finalized outputs
    write_run_info(cfg, group, target_dir)

    log_file = logging.FileHandler(target_dir + "OUT/KARP_log.txt", mode="w")
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(log_file)
    start = time.perf_counter()
    try:
        log.info(f"KARP reducing {group.objid}")
        log.debug(group.describe())
        graph = grapher(cfg, group.objid, target_dir)
        sci_mods = sci_modules.sci_modules(cfg, group, target_dir, graph)
        scinums = sorted(group.science)

        if steps["reduce"]:
            masterbias, final_flat = sci_tools.get_cal_images(group.bias, group.flat, sci_mods.geom, cfg["flat"], graph)
            #individual image processing
            for scinum in scinums:
                sci_mods.reduce_image(scinum, masterbias, final_flat, verbose)

        # Scale and combine the extracted spectra OUT/<objid>_<n>_extracted.fits
        # (read from disk, so this also works on an earlier run's output)
        sci_mods.snr = None
        if steps["combine"]:
            sci_mods.snr = sci_mods.combine_spectra(scinums, verbose, jackknife=cfg["combine"]["jackknife"])

        # Optional: continuum-normalize the combined spectrum (OUT/<objid>_combined.fits)
        if steps["normalize"]:
            sci_mods.normalize_combined(verbose)
        log.info(f"{group.objid} took {timedelta(seconds=time.perf_counter() - start)}")
    except Exception:
        log.exception(f"KARP stopped with an error while reducing {group.objid}")
        raise
    finally:
        log.removeHandler(log_file)
        log_file.close()
    return sci_mods

def main(argv=None, overrides=None):
    """
    Run KARP. `argv` are command-line arguments (default: sys.argv); `overrides` is an
    optional {"section.key": value} dict applied on top of the config file.
    Returns {output name: sci_modules instance} for the groups that were reduced.
    """
    parser = argparse.ArgumentParser(prog="KARP", description="KOSMOS Astronomical Reduction Pipeline")
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

    # Let the user know the program has started
    log.info("KARP is swimming! (Code is running)")
    for i in range(6):
        log.info("- " * i + "><(((º>")
    starttime = time.perf_counter()

    results = {}
    for group in groups:
        log.info(f"\n=== {group.objid} ===")
        results[group.out_name] = run_group(cfg, group)

    #play finishing sound
    if cfg["output"]["sound"]:
        play_finish_sound()

    log.info(f"KARP took: {timedelta(seconds=time.perf_counter() - starttime)} to run")
    log.info("><(((º>")
    return results

#------------------------------------

if __name__ == "__main__":
    main()
