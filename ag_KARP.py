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
from scipy.optimize import differential_evolution
from grapher import grapher
import numpy as np
import argparse
import json
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

def write_run_info(cfg, group, target_dir):
    """Record the settings and frames used, for provenance."""
    with open(target_dir + "OUT/KARP_run_info.txt", "w") as f:
        f.write(group.describe() + "\n\n")
        f.write("Settings:\n")
        f.write(json.dumps(cfg, indent=2) + "\n")

def run_group(cfg, group):
    """Reduce one object/setup group. Returns its sci_modules instance."""
    steps = cfg["steps"]
    verbose = cfg["output"]["verbose"]
    optimize = steps["optimize"]
    if not steps["reduce"] and (steps["normalize"] or optimize):
        raise ConfigError("[steps] normalize and optimize need reduce = true in the same run "
                          "(reduced spectra are not reloaded from disk yet)")

    target_dir = os.path.join(cfg["paths"]["output_dir"], group.out_name) + "/"
    os.makedirs(target_dir + "OUT", exist_ok=True) # Finalized outputs
    write_run_info(cfg, group, target_dir)

    graph = grapher(cfg, group.objid, target_dir)
    sci_mods = sci_modules.sci_modules(cfg, group, target_dir, graph)
    scinums = sorted(group.science)

    def process_images(x=None):
        #optimize overrides the aperture and normalization settings
        if x is not None:
            sci_mods.appw = int(round(x[0]))
            graph.appw = int(round(x[0]))
            sci_mods.norm_line_width = int(round(x[1] * 5))
            sci_mods.norm_line_boxcar = int(round(x[2] * 5))

        SNR = None # Only set when spectra are combined
        if steps["reduce"] or optimize:
            masterbias, final_flat = sci_tools.get_cal_images(group.bias, group.flat, sci_mods.geom, verbose, graph)
            #individual image processing
            for scinum in scinums:
                sci_mods.reduce_image(scinum, masterbias, final_flat, verbose)

        if steps["normalize"] or optimize:
            for scinum in scinums:
                sci_mods.normalization(scinum, verbose, optimize)

        # Median-combine the normalized spectra OUT/<objid>_<n>_OUT.fits of this group's frames
        if steps["combine"] or optimize:
            SNR = sci_mods.combine_spectra(scinums, verbose, jackknife=cfg["combine"]["jackknife"])

        if optimize:
            print(f"SNR {SNR}, APPW: {(sci_mods.appw*2)+1}, NormW: {sci_mods.norm_line_width}, NormBox: {sci_mods.norm_line_boxcar}")
        return SNR

    if optimize:
        bounds = [
            (10, 20),    # a_width
            (130 / 5, 250 / 5),    # norm_line_width
            (50 / 5, 100 / 5),    # norm_line_boxcar
            ] # division by 5 is to reduce parameter space and speed up optimization
        res = differential_evolution(lambda x: -process_images(x), bounds=bounds, maxiter=15, strategy='best1bin', popsize=4, tol=8, atol=10, workers=1, mutation=(0.2, 0.5))
        print(f"Optimal aperture width: {int(round(res.x[0])) * 2 + 1}, SNR: {-res.fun}")
        print(f"Optimal norm width: {res.x[1] * 5} and optimal boxcar: {res.x[2] * 5}")
        sci_mods.snr = -res.fun
    else:
        sci_mods.snr = process_images()

    if steps["fit_lines"]:
        radial_vel = sci_tools.fit_vel_lines(sci_mods.gwave, sci_mods.med_comb, graph)
        sci_tools.fit_metal_lines(sci_mods.gwave, sci_mods.med_comb, sci_mods.sig_final, cfg["lines"]["metal_mask"], radial_vel, graph)
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

    groups, problems = discover(cfg)
    print(f"KARP found {len(groups)} object(s) to reduce in {cfg['paths']['raw_dir']}:")
    for g in groups:
        print(g.describe())
    for p in problems:
        print("Skipping:", p)
    if args.list or not groups:
        return {}

    # Let the user know the program has started
    print("KARP is swimming! (Code is running)")
    for i in range(6):
        print("- " * i + "><(((º>")
    starttime = time.perf_counter()

    results = {}
    for group in groups:
        print(f"\n=== {group.objid} ===")
        results[group.out_name] = run_group(cfg, group)

    # Duration_run is how long KARP took to run
    duration_run = timedelta(seconds=time.perf_counter()-starttime)
    for name in results:
        with open(os.path.join(cfg["paths"]["output_dir"], name, "OUT", "KARP_OUT_log.txt"), "a") as f:
            f.write(f"KARP took: {duration_run} to run\n")
            f.write("><(((º>\n")

    #play finishing sound
    if cfg["output"]["sound"]:
        play_finish_sound()

    print("KARP took:",duration_run," to run")
    print("><(((º>")
    return results

#------------------------------------

if __name__ == "__main__":
    main()
