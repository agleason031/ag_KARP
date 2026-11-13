"""
Make a reference wavelength solution for a grating (karp/data/wavecal/kosmos_<disperser>.json)
from a table of identified arc lines: a KARP wavecal table (diagnostics/frame_<n>/<obj>_wavecal_<n>.txt)
or any table with columns row, wavelength, -, -, -, used (1 = use the line).

    python scripts/make_wavecal_reference.py TABLE --disperser red --binning-y 1 --order 5 \\
        --lamps ARGON -o karp/data/wavecal/kosmos_red.json

KARP starts each night's line identification from this solution, so it only needs to be
roughly right (a shift of up to [wavecal] search rows is found automatically).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from karp.wavecal import format_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table")
    ap.add_argument("--disperser", required=True)
    ap.add_argument("--binning-y", type=int, required=True, help="row (dispersion-axis) binning of that frame")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--lamps", nargs="+", default=["ARGON"])
    ap.add_argument("--source", default="")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    data = np.loadtxt(args.table, comments="#")
    used = data[:, 5].astype(bool)
    lines = [[round(float(r), 3), float(l)] for r, l in data[used][:, :2]]
    ref = {"disperser": args.disperser, "binning_y": args.binning_y, "order": args.order,
           "lamps": args.lamps, "source": args.source or f"made from {args.table}", "lines": lines}
    with open(args.output, "w") as f:
        f.write(format_reference(ref))
    print(f"Wrote {args.output} with {len(lines)} lines")


if __name__ == "__main__":
    main()
