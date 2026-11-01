"""Populate the monitoring dashboard: normal traffic, then a deliberately shifted cohort (Locust).

    python scripts/run.py simulate                                    # local API on :8000
    python scripts/run.py simulate --host https://<app-runner-url> --users 5 --seconds 90

Phase 1 replays real customers (``BOREAL_COHORT=normal``), phase 2 sends the shifted cohort. After
each phase the API's ``/monitoring/report`` is saved, so the before/after traffic lights are
evidence, not a screenshot. Outputs: ``reports/loadtest/<phase>_stats.csv``, ``<phase>.html``
(Locust's own report), ``reports/metrics/monitoring_<phase>.json`` and the alert history
``reports/metrics/monitoring_alerts.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def report(host: str) -> dict:  # type: ignore[type-arg]
    with urllib.request.urlopen(f"{host}/monitoring/report?fresh=true", timeout=60) as r:  # noqa: S310
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=10)
    ap.add_argument("--seconds", type=int, default=120, help="duration of each phase")
    ap.add_argument("--phases", default="normal,shifted")
    a = ap.parse_args()
    out = ROOT / "reports" / "loadtest"
    out.mkdir(parents=True, exist_ok=True)
    host = a.host.rstrip("/")
    for phase in a.phases.split(","):
        print(f"\n=== phase '{phase}': {a.users} users for {a.seconds}s against {host}", flush=True)
        cmd = [
            sys.executable, "-m", "locust", "-f", str(ROOT / "loadtest" / "locustfile.py"),
            "--headless", "-u", str(a.users), "-r", str(max(1, a.users // 2)),
            "-t", f"{a.seconds}s", "--host", host, "--only-summary",
            "--csv", str(out / phase), "--html", str(out / f"{phase}.html"),
        ]  # fmt: skip
        code = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, BOREAL_COHORT=phase))
        if code not in (0, 1):  # locust exits 1 when any request failed; the report shows which
            sys.exit(f"locust failed with exit code {code}")
        rep = report(host)
        (ROOT / "reports" / "metrics" / f"monitoring_{phase}.json").write_text(
            json.dumps(rep, indent=2, default=str) + "\n", encoding="utf-8"
        )
        lights = ", ".join(f"{c['name']}={c['status']}" for c in rep["checks"])
        print(f"after '{phase}': overall={rep['status']} | {lights}", flush=True)
    with urllib.request.urlopen(f"{host}/monitoring/alerts?limit=100", timeout=60) as r:  # noqa: S310
        alerts = json.loads(r.read())
    (ROOT / "reports" / "metrics" / "monitoring_alerts.json").write_text(
        json.dumps(alerts, indent=2) + "\n", encoding="utf-8"
    )
    print(f"alert history: {len(alerts)} level changes saved", flush=True)


if __name__ == "__main__":
    main()
