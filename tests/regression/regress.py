#!/usr/bin/env python3
"""
Regression harness for KARP.

Runs the full pipeline (ag_KARP.py) on the G093440 test set, captures the
intermediate and final arrays, and either saves them as the reference or
compares them against the saved reference.

    python tests/regression/regress.py make-reference
    python tests/regression/regress.py compare

The test FITS frames are expected in test_data/in/ (not tracked by git).
Each run writes its full output (plots, FITS) to tests/regression/runs/<timestamp>/.
"""
import argparse
import datetime
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARAMS = HERE / "G093440.kparam"
REF_DIR = HERE / "reference"
REF_NPZ = REF_DIR / "G093440_reference.npz"
REF_META = REF_DIR / "G093440_reference.json"
RUNS = HERE / "runs"
DATA = REPO / "test_data" / "in"

# Tolerances for calling an array "unchanged".
RTOL = 1e-6
ATOL = 1e-9


# ---------------------------------------------------------------------------
# running the pipeline
# ---------------------------------------------------------------------------

def write_run_params(run_dir):
    """Copy the parameter file with dloc and target_dir pointed at absolute paths."""
    lines = []
    for line in PARAMS.read_text().splitlines():
        key = line.split("=", 1)[0].strip()
        if key == "dloc":
            line = f"dloc = {DATA}/"
        elif key == "target_dir":
            line = f"target_dir = {run_dir}/"
        lines.append(line)
    path = run_dir / "run.kparam"
    path.write_text("\n".join(lines) + "\n")
    return path


def run_pipeline():
    """Run ag_KARP.py as __main__ and return (its globals, run directory)."""
    if not DATA.is_dir():
        sys.exit(f"Test data not found at {DATA}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS / stamp
    run_dir.mkdir(parents=True)
    kparam = write_run_params(run_dir)

    sys.path.insert(0, str(REPO))
    old_argv, old_cwd = sys.argv, os.getcwd()
    sys.argv = ["ag_KARP.py", "--kp", str(kparam)]
    os.chdir(REPO)
    try:
        g = runpy.run_path(str(REPO / "ag_KARP.py"), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return g, run_dir


def block_mean(img, by=64, bx=16):
    """Downsample a 2D image by averaging blocks, for a compact fingerprint."""
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    ny2, nx2 = ny - ny % by, nx - nx % bx
    return img[:ny2, :nx2].reshape(ny2 // by, by, nx2 // bx, bx).mean(axis=(1, 3))


def snapshot(g, run_dir):
    """Collect the arrays worth tracking from a finished run."""
    from astropy.table import Table
    import sci_tools

    sm = g["sci_mods"]
    snap = {}

    # Calibration frames: stats + block-averaged thumbnails (full frames are 37 MB each).
    masterbias, final_flat = sci_tools._calibration_cache[sm.objid]
    for name, img in (("masterbias", np.asarray(masterbias)), ("final_flat", np.asarray(final_flat))):
        snap[f"{name}_thumb"] = block_mean(img)
        snap[f"{name}_stats"] = np.array([np.nanmedian(img), np.nanmean(img), np.nanstd(img)])

    snap["v_helio"] = np.array([sm.v_helio])

    for n in sorted(sm.cen_line_dict):
        snap[f"cen_line_{n}"] = np.asarray(sm.cen_line_dict[n])
        snap[f"arc_centers_{n}"] = np.array([fit[1] for fit in sm.lam_fit_dict[n]])
        snap[f"flux_raw_{n}"] = np.asarray(sm.flux_raw_cor1_dict[n])
        snap[f"sky_raw_{n}"] = np.asarray(sm.sky_raw_dict[n])
        snap[f"wavelengths_{n}"] = np.asarray(sm.wavelengths_dict[n])
        out = run_dir / "OUT" / f"{sm.objid}_{n}_OUT.fits"
        if out.exists():
            t = Table.read(out)
            snap[f"norm_lam_{n}"] = np.asarray(t["Lam"])
            snap[f"norm_flux_{n}"] = np.asarray(t["Flux"])
            snap[f"norm_sigma_{n}"] = np.asarray(t["Sigma"])

    snap["comb_lam"] = np.asarray(sm.gwave)
    snap["comb_flux"] = np.asarray(sm.med_comb)
    snap["comb_sigma"] = np.asarray(sm.sig_final)
    snap["snr"] = np.array([g.get("SNR", np.nan)])
    return snap


def git_info():
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

# (wavelength key, value keys) pairs compared on the reference wavelength grid
SPECTRA = [
    ("wavelengths_{n}", ["flux_raw_{n}", "sky_raw_{n}"]),
    ("norm_lam_{n}", ["norm_flux_{n}", "norm_sigma_{n}"]),
]


def array_diff(ref, new):
    if ref.shape != new.shape:
        return "CHANGED", f"shape {ref.shape} -> {new.shape}"
    ref_f, new_f = ref.astype(float), new.astype(float)
    nan_mismatch = np.sum(np.isnan(ref_f) != np.isnan(new_f))
    ok = np.isfinite(ref_f) & np.isfinite(new_f)
    if not ok.any():
        return ("same" if nan_mismatch == 0 else "CHANGED"), "all NaN"
    d = np.abs(new_f[ok] - ref_f[ok])
    scale = np.abs(ref_f[ok])
    same = nan_mismatch == 0 and np.all(d <= ATOL + RTOL * scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.nanmax(np.where(scale > 0, d / scale, 0))
    detail = f"max|d|={d.max():.3g}  max rel={rel:.3g}"
    if nan_mismatch:
        detail += f"  NaN mismatches={nan_mismatch}"
    return ("same" if same else "CHANGED"), detail


def spectrum_diff(ref_lam, ref_val, new_lam, new_val):
    """Interpolate the new spectrum onto the reference wavelengths and compare."""
    def clean(lam, val):
        m = np.isfinite(lam) & np.isfinite(val)
        lam, val = lam[m], val[m]
        order = np.argsort(lam)
        return lam[order], val[order]
    rl, rv = clean(ref_lam, ref_val)
    nl, nv = clean(new_lam, new_val)
    if len(rl) < 2 or len(nl) < 2:
        return "n/a"
    m = (rl >= nl[0]) & (rl <= nl[-1])
    interp = np.interp(rl[m], nl, nv)
    d = np.abs(interp - rv[m])
    scale = np.nanmedian(np.abs(rv[m])) or 1.0
    return (f"on ref grid: median|d|/median|ref|={np.median(d) / scale:.3g}  "
            f"p99={np.percentile(d, 99) / scale:.3g}  overlap {rl[m][0]:.1f}-{rl[m][-1]:.1f} A")


def compare(ref, new):
    changed = 0
    keys = sorted(set(ref) | set(new))
    width = max(len(k) for k in keys)
    print(f"\n{'array':<{width}}  status   detail")
    print("-" * (width + 60))
    for k in keys:
        if k not in ref:
            print(f"{k:<{width}}  NEW      shape {new[k].shape}")
            changed += 1
            continue
        if k not in new:
            print(f"{k:<{width}}  MISSING")
            changed += 1
            continue
        status, detail = array_diff(ref[k], new[k])
        changed += status != "same"
        print(f"{k:<{width}}  {status:<7}  {detail}")

    # Spectra: also compare on a common wavelength grid, since pixel-by-pixel
    # comparison is meaningless once the wavelength solution or pixel set changes.
    print("\nSpectra resampled onto the reference wavelength grid:")
    nums = sorted({k.rsplit("_", 1)[1] for k in ref if k.startswith("cen_line_")})
    pairs = [(lk.format(n=n), vk.format(n=n)) for n in nums for lk, vks in SPECTRA for vk in vks]
    pairs += [("comb_lam", "comb_flux"), ("comb_lam", "comb_sigma")]
    for lk, vk in pairs:
        if all(x in ref and x in new for x in (lk, vk)):
            print(f"  {vk:<{width}}  {spectrum_diff(ref[lk], ref[vk], new[lk], new[vk])}")
    return changed


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["make-reference", "compare"])
    ap.add_argument("--note", default="", help="reason for a new reference (stored in its metadata)")
    args = ap.parse_args()

    g, run_dir = run_pipeline()
    snap = snapshot(g, run_dir)
    np.savez_compressed(run_dir / "snapshot.npz", **snap)

    if args.command == "make-reference":
        REF_DIR.mkdir(exist_ok=True)
        np.savez_compressed(REF_NPZ, **snap)
        meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "git": git_info(), "note": args.note,
                "snr": float(snap["snr"][0]), "v_helio": float(snap["v_helio"][0])}
        REF_META.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nReference written to {REF_NPZ}")
        print(json.dumps(meta, indent=2))
        return 0

    if not REF_NPZ.exists():
        sys.exit("No reference yet; run make-reference first.")
    ref = dict(np.load(REF_NPZ))
    print(f"Reference: {json.loads(REF_META.read_text())}")
    print(f"This run:  {git_info()}  output in {run_dir}")
    n = compare(ref, snap)
    print(f"\n{n} array(s) changed." if n else "\nAll arrays match the reference.")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
