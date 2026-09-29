"""Thin wrapper around the data-quality assertion suite. Prefer `make dq`."""

import sys

from boreal_marche.data.quality import main

if __name__ == "__main__":
    sys.exit(main())
