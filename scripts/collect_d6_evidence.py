"""Collect what the D6 documents cite about the system *as it runs now* into
``reports/metrics/d6_evidence.json``.

    python scripts/collect_d6_evidence.py                 # live check + tests and coverage
    python scripts/collect_d6_evidence.py --skip-tests    # live check only
    python scripts/collect_d6_evidence.py --skip-tests --presentation-video-url https://youtu.be/…

* the live API: ``/version`` (deployed commit, model version) and ``/ready``, with the time checked;
* the live interface: HTTP status of the Streamlit URL (a sleeping app answers 303 to its wake-up
  page, which still means "deployed");
* test count and line coverage from a real ``pytest --cov`` run (same runner as the D4 collector);
* the annotated tags present in the repository;
* when given, the recorded D6 presentation (URL and the title YouTube reports for it).
Nothing is typed in by hand: re-run it the morning of the presentation and rebuild D6.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "d6_evidence.json"
D4 = json.loads((ROOT / "reports" / "metrics" / "d4_evidence.json").read_text("utf-8"))
sys.path.insert(0, str(ROOT / "scripts"))
from collect_d4_evidence import tests  # noqa: E402


def get(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "boreal-d6-evidence"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310 - fixed https URLs
            return r.status, r.read()
    except urllib.error.HTTPError as e:  # 3xx/4xx/5xx are evidence too
        return e.code, b""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=D4["live"]["api_url"])
    ap.add_argument("--ui-url", default=D4["live"]["ui_url"])
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--presentation-video-url", help="the recorded D6 talk (YouTube)")
    a = ap.parse_args()
    ev = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017
    api = a.api_url.rstrip("/")
    _, version = get(f"{api}/version")
    ready_status, _ = get(f"{api}/ready")
    ui_status, _ = get(a.ui_url)
    ev["live"] = {
        "api_url": api,
        "ui_url": a.ui_url.rstrip("/"),
        "version": json.loads(version) if version else None,
        "ready_status": ready_status,
        "ui_status": ui_status,
        "checked_at": now,
    }
    if not a.skip_tests:
        ev["tests"] = tests()
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    ev["tags"] = tags.stdout.split()
    if a.presentation_video_url:
        q = urllib.parse.urlencode({"url": a.presentation_video_url, "format": "json"})
        _, meta = get(f"https://www.youtube.com/oembed?{q}")
        ev["presentation_video"] = {
            "url": a.presentation_video_url,
            "title": json.loads(meta)["title"] if meta else None,
            "checked_at": now,
        }
    ev["collected_at"] = now
    OUT.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ev, indent=2))


if __name__ == "__main__":
    main()
