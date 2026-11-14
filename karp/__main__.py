"""python -m karp config.toml [--list]"""
import sys

from .cli import main

sys.exit(main())
