"""Collect the measured numbers the D4 note cites into ``reports/metrics/d4_evidence.json``.

    python scripts/collect_d4_evidence.py                       # images + tests + corpus
    python scripts/collect_d4_evidence.py --api-url https://… --ui-url https://… --video-url https://…

* image sizes: compressed (``docker image inspect``, = pushed to ECR) and unpacked (``du`` inside
  the image) for the ``--images`` tags; ``boreal-api:v0.3`` (built from the v0.3 tag) is "before";
* test count and line coverage from a real ``pytest --cov`` run (hashing embedder, as in CI);
* corpus size; and, when given, the live URLs with the commit their ``/version`` reports right now.
Nothing is typed in by hand: re-run it and the report numbers follow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "d4_evidence.json"


def image_size(tag: str) -> dict[str, int] | None:
    """Compressed size (what is pushed to ECR) and unpacked filesystem size, in bytes / MB."""
    r = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Size}}"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    du = subprocess.run(
        ["docker", "run", "--rm", "--user", "root", "--entrypoint", "sh", tag, "-c",
         "du -sxm / 2>/dev/null | cut -f1"],
        capture_output=True, text=True,
    )  # fmt: skip
    return {"compressed_bytes": int(r.stdout.strip()), "unpacked_mb": int(du.stdout.strip() or 0)}


def tests() -> dict[str, float]:
    env = dict(os.environ, BOREAL_EMBEDDER="hash", BOREAL_MONITOR_LOOP="0", PYTHONUTF8="1")
    env["PYTHONPATH"] = str(ROOT / "src")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:warnings", "--cov=boreal_marche",
         "--cov-report=term", "-o", "addopts="],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )  # fmt: skip
    out = r.stdout
    passed = re.search(r"(\d+) passed", out)
    skipped = re.search(r"(\d+) skipped", out)
    total = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", out, flags=re.M)
    if r.returncode != 0:
        sys.exit("tests failed — fix them before collecting evidence\n" + out[-2000:])
    return {
        "passed": int(passed.group(1)) if passed else 0,
        "skipped": int(skipped.group(1)) if skipped else 0,
        "coverage_pct": int(total.group(1)) if total else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="boreal-api:v0.3,boreal-api:d4,boreal-ui:d4")
    ap.add_argument("--api-url")
    ap.add_argument("--ui-url")
    ap.add_argument("--video-url")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    ev = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    ev["collected_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017
    sizes = {t: image_size(t) for t in a.images.split(",")}
    if any(sizes.values()):
        ev["images"] = {t: s for t, s in sizes.items() if s}
    if not a.skip_tests:
        ev["tests"] = tests()
    corpus = sorted((ROOT / "docs" / "corpus").glob("*.md"))
    ev["corpus"] = {
        "documents": len(corpus),
        "words": sum(len(p.read_text("utf-8").split()) for p in corpus),
    }
    if a.api_url:
        with urllib.request.urlopen(f"{a.api_url.rstrip('/')}/version", timeout=60) as r:  # noqa: S310
            version = json.loads(r.read())
        ev["live"] = {
            "api_url": a.api_url.rstrip("/"),
            "ui_url": (a.ui_url or "").rstrip("/") or None,
            "version": version,
            "checked_at": ev["collected_at"],
        }
    if a.video_url:
        ev["video_url"] = a.video_url
    OUT.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()
