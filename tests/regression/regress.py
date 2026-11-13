#!/usr/bin/env python3
"""
Regression harness for KARP.

Runs the full pipeline (karp) on the G093440 test set, captures the
intermediate and final arrays, and either saves them as the reference or
compares them against the saved reference.

    python tests/regression/regress.py make-reference
    python tests/regression/regress.py compare

The test FITS frames are expected in test_data/in/ (not tracked by git).
Each run writes its full output (plots, FITS) to tests/regression/runs/<timestamp>/G093440/.
Settings are in G093440.toml; --set section.key=value overrides one for a single run.
"""
import argparse
import datetime
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "G093440.toml"
OBJECT = "G093440"
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

def parse_override(kv):
    """'section.key=value' with a TOML value (e.g. cosmic_rays.enabled=false) -> (key, value)."""
    key, value = kv.split("=", 1)
    if "." not in key:
        sys.exit(f"--set needs section.key=value, got {kv!r}")
    return key.strip(), tomllib.loads(f"v = {value}")["v"]


def run_pipeline(overrides=None):
    """Run KARP on the test set and return (its ObjectReduction, output dir for the object)."""
    if not DATA.is_dir():
        sys.exit(f"Test data not found at {DATA}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS / stamp
    run_dir.mkdir(parents=True)

    sys.path.insert(0, str(REPO))
    from karp import cli
    settings = {"paths.raw_dir": str(DATA), "paths.output_dir": str(run_dir)}
    settings.update(overrides or {})
    results = cli.main([str(CONFIG)], overrides=settings)
    if list(results) != [OBJECT]:
        sys.exit(f"Expected one object {OBJECT}, got {list(results)}")
    return results[OBJECT], run_dir / OBJECT


def block_mean(img, by=64, bx=16):
    """Downsample a 2D image by averaging blocks, for a compact fingerprint."""
    img = np.asarray(img, dtype=float)
    ny, nx = img.shape
    ny2, nx2 = ny - ny % by, nx - nx % bx
    return img[:ny2, :nx2].reshape(ny2 // by, by, nx2 // bx, bx).mean(axis=(1, 3))


def snapshot(obj, out_dir):
    """Collect the arrays worth tracking from a finished run (obj: karp.pipeline.ObjectReduction)."""
    from karp import calibration

    snap = {}

    # Calibration frames: stats + block-averaged thumbnails (full frames are 37 MB each).
    # A run makes or loads exactly one master bias / flat pair
    (masterbias, final_flat), = calibration.cached_calibrations()
    for name, img in (("masterbias", np.asarray(masterbias)), ("final_flat", np.asarray(final_flat))):
        snap[f"{name}_thumb"] = block_mean(img)
        snap[f"{name}_stats"] = np.array([np.nanmedian(img), np.nanmean(img), np.nanstd(img)])

    for n, fr in sorted(obj.frames.items()):
        snap[f"v_helio_{n}"] = np.array([fr.v_helio])
        snap[f"cen_line_{n}"] = np.asarray(fr.trace_columns)
        sol = fr.solution
        snap[f"arc_centers_{n}"] = sol.rows[sol.used]
        snap[f"arc_lam_{n}"] = sol.lam[sol.used]
        snap[f"arc_rms_kms_{n}"] = np.array([sol.rms_kms])
        if obj.wcfg["flexure"] != "off":
            snap[f"sky_offsets_{n}"] = np.array([[l, fr.flexure.get(l, np.nan)] for l in obj.sky_lines])
        snap[f"flux_raw_{n}"] = np.asarray(fr.flux)
        snap[f"sky_raw_{n}"] = np.asarray(fr.sky)
        snap[f"err_raw_{n}"] = np.asarray(fr.sigma)
        snap[f"wavelengths_{n}"] = np.asarray(fr.wave)

    # Combined spectrum (not normalized)
    if obj.combined is not None:
        snap["comb_lam"] = np.asarray(obj.combined.wave)
        snap["comb_flux"] = np.asarray(obj.combined.flux)
        snap["comb_sigma"] = np.asarray(obj.combined.sigma)
        snap["comb_mask"] = np.asarray(obj.combined.mask)
    # Normalized combined spectrum (optional step)
    if obj.normalized is not None:
        nm = obj.normalized
        snap["norm_lam"] = np.asarray(nm.wave)
        snap["norm_flux"] = np.asarray(nm.flux)
        snap["norm_sigma"] = np.asarray(nm.sigma)
        snap["norm_continuum"] = np.asarray(nm.continuum)
        snap["norm_contfit"] = np.asarray(nm.used, dtype=np.int16)
        snap["norm_mask"] = np.asarray(nm.mask)
    snap["snr"] = np.array([obj.snr if obj.snr is not None else np.nan])
    return snap


def git_info():
    def run(*cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("git", "rev-parse", "--short", "HEAD"),
            # The reference files themselves don't count as uncommitted code changes
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no",
                              "--", ".", ":!tests/regression/reference"))}


# ---------------------------------------------------------------------------
# comparing
# ---------------------------------------------------------------------------

# (wavelength key, value keys) pairs compared on the reference wavelength grid
SPECTRA = [
    ("wavelengths_{n}", ["flux_raw_{n}", "sky_raw_{n}", "err_raw_{n}"]),
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
    pairs += [("comb_lam", "comb_flux"), ("comb_lam", "comb_sigma"),
              ("norm_lam", "norm_flux"), ("norm_lam", "norm_sigma"), ("norm_lam", "norm_continuum")]
    for lk, vk in pairs:
        if all(x in ref and x in new for x in (lk, vk)):
            print(f"  {vk:<{width}}  {spectrum_diff(ref[lk], ref[vk], new[lk], new[vk])}")
    return changed


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["make-reference", "compare"])
    ap.add_argument("--note", default="", help="reason for a new reference (stored in its metadata)")
    ap.add_argument("--ref", type=Path, default=None,
                    help="compare against this snapshot.npz (e.g. an earlier run) instead of the saved reference")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override a setting for this run, value in TOML syntax (repeatable), "
                         "e.g. --set cosmic_rays.enabled=false")
    args = ap.parse_args()
    overrides = dict(parse_override(kv) for kv in args.set)
    if overrides and args.command == "make-reference":
        sys.exit("--set is for experiments; edit G093440.toml to change the reference settings.")

    obj, out_dir = run_pipeline(overrides)
    snap = snapshot(obj, out_dir)
    np.savez_compressed(out_dir / "snapshot.npz", **snap)
    run_dir = out_dir

    if args.command == "make-reference":
        REF_DIR.mkdir(exist_ok=True)
        np.savez_compressed(REF_NPZ, **snap)
        meta = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "git": git_info(), "note": args.note,
                "snr": float(snap["snr"][0]),
                "v_helio": {k[len("v_helio_"):]: float(v[0]) for k, v in snap.items() if k.startswith("v_helio_")}}
        REF_META.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"\nReference written to {REF_NPZ}")
        print(json.dumps(meta, indent=2))
        return 0

    if args.ref is not None:
        ref = dict(np.load(args.ref))
        print(f"Reference: {args.ref}")
    else:
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
