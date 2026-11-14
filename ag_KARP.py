#!/usr/bin/env python3
"""
KARP: KOSMOS Astronomical Reduction Pipeline.

    python ag_KARP.py config.toml          # reduce every object found
    python ag_KARP.py config.toml --list   # only show which frames would be used

Same as `python -m karp` or the `karp` command; the code is in the karp/ package.
"""
import sys

from karp.cli import main

if __name__ == "__main__":
    sys.exit(main())
