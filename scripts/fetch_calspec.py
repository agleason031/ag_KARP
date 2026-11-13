"""
Download spectrophotometric standards from HST CALSPEC and write them as KARP text
files (karp/data/standards/<name>.txt) plus an index (index.json) of names and aliases.

    python scripts/fetch_calspec.py              # the standards listed below
    python scripts/fetch_calspec.py --only g191b2b

Each output file has columns: air wavelength (A), flux (erg/s/cm^2/A), FWHM of the
CALSPEC spectrum (A), trimmed to 3000-11000 A. CALSPEC wavelengths are in vacuum; they
are converted to air (Morton 2000, the IAU standard) to match KARP's wavelengths.
To add a standard, add a line to STANDARDS (file name from the CALSPEC directory).
"""
import argparse
import json
import os
import urllib.request

import numpy as np
from astropy.io import fits

URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "karp", "data", "standards")
LAM_MIN, LAM_MAX = 3000.0, 11000.0

# name: (CALSPEC file, aliases as they might appear in OBJNAME)
STANDARDS = {
    "g191b2b": ("g191b2b_stiswfcnic_004.fits", ["G191-B2B", "G191B2B", "BD+52 913"]),
    "bd28d4211": ("bd_28d4211_stis_005.fits", ["BD+28 4211", "BD+28D4211", "BD28"]),
    "feige34": ("feige34_stis_006.fits", ["Feige 34", "F34"]),
    "feige110": ("feige110_stisnic_008.fits", ["Feige 110", "F110"]),
    "hz44": ("hz44_stis_006.fits", ["HZ 44"]),
    "hz43": ("hz43_stis_005.fits", ["HZ 43", "HZ43A"]),
    "hz4": ("hz4_stis_009.fits", ["HZ 4"]),
    "hz2": ("hz2_stis_001.fits", ["HZ 2"]),
    "gd71": ("gd71_stiswfcnic_004.fits", ["GD 71"]),
    "gd153": ("gd153_stiswfcnic_004.fits", ["GD 153"]),
    "bd17d4708": ("bd_17d4708_stisnic_007.fits", ["BD+17 4708", "BD+17D4708"]),
    "agk81d266": ("agk_81d266_stisnic_007.fits", ["AGK+81 266", "AGK+81D266"]),
    "grw70d5824": ("grw_70d5824_stiswfcnic_003.fits", ["GRW+70 5824", "GRW+70D5824"]),
    "lds749b": ("lds749b_stisnic_008.fits", ["LDS 749B"]),
    "hd93521": ("hd93521_stis_007.fits", ["HD 93521"]),
}


def vac_to_air(lam):
    """Vacuum to air wavelength (A), Morton (2000); unchanged below 2000 A."""
    s2 = (1e4 / lam) ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return np.where(lam > 2000, lam / n, lam)


def fetch(name, fname):
    req = urllib.request.Request(URL + fname, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    tmp = os.path.join(HERE, f".{fname}")
    with open(tmp, "wb") as f:
        f.write(raw)
    try:
        with fits.open(tmp) as h:
            d = h[1].data
            lam = np.asarray(d["WAVELENGTH"], float)
            flux = np.asarray(d["FLUX"], float)
            fwhm = np.asarray(d["FWHM"], float) if "FWHM" in d.names else np.zeros_like(lam)
            target = str(h[0].header.get("TARGETID", name)).strip()
    finally:
        os.remove(tmp)
    m = (lam >= LAM_MIN) & (lam <= LAM_MAX) & np.isfinite(flux) & (flux > 0)
    out = os.path.join(HERE, f"{name}.txt")
    with open(out, "w") as f:
        f.write(f"# {target}: HST CALSPEC {fname} ({URL}{fname})\n")
        f.write("# Converted to air wavelengths (Morton 2000); trimmed to "
                f"{LAM_MIN:.0f}-{LAM_MAX:.0f} A.\n")
        f.write("# air_wavelength_A  flux_erg/s/cm2/A  fwhm_A\n")
        for l, fl, fw in zip(vac_to_air(lam[m]), flux[m], fwhm[m]):
            f.write(f"{l:.3f} {fl:.5e} {fw:.2f}\n")
    print(f"{name}: {m.sum()} points -> {os.path.relpath(out)}")
    return {"file": f"{name}.txt", "aliases": STANDARDS[name][1], "source": f"CALSPEC {fname}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", help="fetch only these standards")
    args = ap.parse_args()
    index_path = os.path.join(HERE, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    for name in args.only or STANDARDS:
        index[name] = fetch(name, STANDARDS[name][0])
    with open(index_path, "w") as f:
        json.dump(dict(sorted(index.items())), f, indent=1)
        f.write("\n")
