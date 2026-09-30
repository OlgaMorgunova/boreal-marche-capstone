"""Print the SHA-256 of the raw workbook (the value recorded in params.yaml and the manifest)."""

import sys
from pathlib import Path

from boreal_marche.config import load_params, repo_root
from boreal_marche.utils import sha256_file

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root() / load_params().data.raw_file
    print(sha256_file(path))
