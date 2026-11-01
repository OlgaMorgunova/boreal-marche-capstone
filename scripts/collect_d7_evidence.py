"""D7 readiness check: every Appendix A line that a script can verify, run against the live system
and the repository, written to ``reports/metrics/d7_evidence.json`` (rendered by the D7 defence
brief). Lines only a person can tick (phone in incognito, a fresh clone on another machine,
rehearsal) are listed as ``manual``.

    python scripts/collect_d7_evidence.py                # live checks + tests and coverage
    python scripts/collect_d7_evidence.py --skip-tests   # live checks only (the defence morning)
    python scripts/collect_d7_evidence.py --skip-tests --defence-video-url https://youtu.be/…

Live: /health, /ready, /version against the head of ``origin/master``, and the three heads end to
end — the golden prediction with its campaign action, /campaign, /recommend firing a rule, /chat
citing a source and refusing an out-of-scope question — then /monitoring/report. Repository:
public visibility and the latest CI and CD runs on ``master`` (GitHub's public API, or
``GITHUB_TOKEN`` while the repository is private), the annotated tags, the §7.2 structure, no raw
data in git, the README video above the fold, and SageMaker's quota request status (AWS CLI,
``--aws-profile``). Nothing is typed in by hand. Exit code 1 if an automated check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "metrics" / "d7_evidence.json"
D4 = json.loads((ROOT / "reports" / "metrics" / "d4_evidence.json").read_text("utf-8"))
REPO = "OlgaMorgunova/boreal-marche-capstone"
TAGS = ["v0.1", "v0.2", "v0.3", "v0.9", "v1.0"]  # course §7.1; v0.9.5 and v0.9.6 are extra
STRUCTURE = [
    "README.md",
    "ARCHITECTURE.md",
    "MODEL_CARD.md",
    "DATA_CARD.md",
    "GOVERNANCE_LOI25.md",
    "Makefile",
    ".env.example",
    "src/boreal_marche",
    "notebooks",
    "tests",
    "docker/compose.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",
    ".github/workflows/retrain.yml",
    "data/raw/online_retail_II.xlsx.dvc",
    "artifacts/model.joblib",
    "artifacts/model_meta.json",
    "artifacts/rules.json",
    "artifacts/index",
    "frontend",
    "docs/corpus",
    "reports",
]
MANUAL = [
    "Application URL opens in an incognito window on a phone",
    "Compose stack tested from a fresh clone on a different machine",
    "Application warm and open in a browser tab before the defence",
    "Fallback video at hand in case the network fails",
    "Prepared: primary metric, baseline and threshold with their justification",
    "Prepared: one answer to 'what would you do differently' and one negative result",
]
sys.path.insert(0, str(ROOT / "scripts"))
from collect_d4_evidence import tests  # noqa: E402


def call(url: str, body: dict[str, Any] | None = None, token: str | None = None) -> tuple[int, Any]:
    headers = {
        "User-Agent": "boreal-d7-evidence",
        "Content-Type": "application/json",
        "X-Traffic-Cohort": "smoke",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers)  # noqa: S310 - fixed URLs
    try:
        with urllib.request.urlopen(req, timeout=90) as r:  # noqa: S310
            raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()[:200]
    except urllib.error.HTTPError as e:  # 3xx/4xx/5xx are evidence too
        return e.code, None
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, str(e)


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip()


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, group: str, item: str, ok: bool | None, detail: Any = "") -> None:
        status = "pending" if ok is None else ("pass" if ok else "fail")
        self.rows.append({"group": group, "item": item, "status": status, "detail": str(detail)})
        print(f"[{status.upper():7s}] {group}: {item} — {str(detail)[:110]}")


def live(c: Checks, api: str, ui: str) -> dict[str, Any]:
    code, h = call(f"{api}/health")
    c.add("live", "/health", code == 200 and h == {"status": "ok"}, code)
    code, r = call(f"{api}/ready")
    c.add("live", "/ready", code == 200 and bool(r and r["ready"]), r["errors"] if r else code)
    _, v = call(f"{api}/version")
    head = git("ls-remote", "origin", "refs/heads/master").split("\t")[0]
    sha = (v or {}).get("git_commit")
    c.add(
        "live",
        "/version = head of master",
        bool(head) and sha == head,
        f"live {str(sha)[:12]} · master {head[:12]} · API {(v or {}).get('api_version')}",
    )
    golden = json.loads((ROOT / "tests" / "golden" / "golden_predictions.json").read_text("utf-8"))
    case = golden["cases"][0]
    code, p = call(f"{api}/predict", {"customer_ref": case["ref"], "features": case["features"]})
    ok = code == 200 and abs(p["p_return"] - case["p_return"]) < 1e-6
    c.add(
        "head A",
        "golden prediction reproduced",
        ok,
        f"p_return={p['p_return']:.6f}" if ok else str(p)[:80],
    )
    c.add(
        "head A",
        "campaign action on /predict",
        ok and p.get("action") is not None,
        p and p.get("action"),
    )
    code, cp = call(f"{api}/campaign")
    c.add("head A", "/campaign list (review + hold-out)", code == 200, cp["counts"] if cp else code)
    code, rec = call(f"{api}/recommend", {"basket": ["22916", "22918"], "k": 3})
    c.add(
        "head B",
        "/recommend fires a rule",
        code == 200 and rec["strategy"].startswith("rules"),
        rec and rec["strategy"],
    )
    code, ch = call(
        f"{api}/chat", {"message": "Combien de temps pour un remboursement ?", "provider": "rule"}
    )
    c.add(
        "head C",
        "/chat answers in French with a citation",
        code == 200 and ch["tool"] == "policy_search" and bool(ch["citations"]),
        ch and f"tool={ch['tool']} lang={ch.get('lang')} citations={len(ch['citations'])}",
    )
    code, oos = call(f"{api}/chat", {"message": "Write me a poem about hockey", "provider": "rule"})
    c.add(
        "head C",
        "/chat refuses out-of-scope",
        code == 200 and bool(oos.get("refused")),
        oos and oos.get("tool"),
    )
    code, mon = call(f"{api}/monitoring/report")
    n = (mon or {}).get("window", {}).get("predictions", 0)
    c.add(
        "monitoring", "dashboard has traffic", code == 200 and n > 0, f"{n} predictions in window"
    )
    code, _ = call(ui)
    c.add("live", "interface reachable (303 = waking up)", code in (200, 303), code)
    return {
        "api_url": api,
        "ui_url": ui,
        "version": v,
        "master_head": head,
        "campaign_counts": cp and cp["counts"],
        "monitoring_checks": {x["name"]: x["status"] for x in (mon or {}).get("checks", [])},
    }


def repository(c: Checks, token: str | None) -> dict[str, Any]:
    gh = f"https://api.github.com/repos/{REPO}"
    code, meta = call(gh, token=token)
    public = code == 200 and meta.get("private") is False
    c.add(
        "repository", "public", public, "public" if public else f"HTTP {code} (private or absent)"
    )
    runs: dict[str, Any] = {}
    for wf in ("ci.yml", "cd.yml"):
        code, r = call(f"{gh}/actions/workflows/{wf}/runs?branch=master&per_page=1", token=token)
        last = (r or {}).get("workflow_runs", [{}])[0] if code == 200 else {}
        runs[wf] = {k: last.get(k) for k in ("head_sha", "conclusion", "html_url", "created_at")}
        c.add(
            "repository",
            f"{wf} green on master",
            None if code != 200 else last.get("conclusion") == "success",
            f"{last.get('conclusion')} @ {str(last.get('head_sha'))[:7]}"
            if last
            else "not readable while private (set GITHUB_TOKEN)",
        )
    for tag in TAGS:
        kind = git("cat-file", "-t", tag)
        c.add(
            "repository",
            f"annotated tag {tag}",
            None if not kind and tag == "v1.0" else kind == "tag",
            kind or "missing",
        )
    missing = [p for p in STRUCTURE if not (ROOT / p).exists()]
    c.add("repository", "§7.2 structure", not missing, missing or f"{len(STRUCTURE)} paths")
    raw = [
        f for f in git("ls-files", "data/raw").splitlines() if not f.endswith((".dvc", ".gitkeep"))
    ]
    c.add("repository", "raw data not committed", not raw, raw or "only the .dvc pointer")
    head = (ROOT / "README.md").read_text("utf-8").splitlines()[:20]
    c.add(
        "repository",
        "README video above the fold",
        any("youtu" in ln for ln in head),
        "link in the first 20 lines",
    )
    return {
        "public": public,
        "runs": runs,
        "tags": git("tag", "--list", "v*").split(),
        "head": git("rev-parse", "HEAD"),
    }


def sagemaker(c: Checks, profile: str) -> dict[str, Any] | None:
    cmd = [
        "aws",
        "service-quotas",
        "list-requested-service-quota-change-history",
        "--service-code",
        "sagemaker",
        "--region",
        "ca-central-1",
        "--profile",
        profile,
        "--output",
        "json",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
        reqs = json.loads(r.stdout)["RequestedQuotas"] if r.returncode == 0 else []
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        reqs = []
    q = next((x for x in reqs if "spot training" in x["QuotaName"]), None)
    if q is None:
        c.add(
            "cloud",
            "SageMaker spot quota",
            None,
            f"not readable (aws sso login --profile {profile})",
        )
        return None
    out = {k: q.get(k) for k in ("QuotaName", "Status", "DesiredValue", "CaseId", "Created")}
    out["Created"] = str(out["Created"])
    c.add(
        "cloud",
        "SageMaker spot quota",
        None if q["Status"] != "APPROVED" else True,
        f"{q['Status']} (case {q.get('CaseId')}, {q['DesiredValue']:.0f} instances)",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default=D4["live"]["api_url"])
    ap.add_argument("--ui-url", default=D4["live"]["ui_url"])
    ap.add_argument("--aws-profile", default="admin-profile")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--defence-video-url", help="the recorded defence (YouTube); kept on re-runs")
    a = ap.parse_args()
    ev = json.loads(OUT.read_text("utf-8")) if OUT.exists() else {}
    c = Checks()
    ev["live"] = live(c, a.api_url.rstrip("/"), a.ui_url.rstrip("/"))
    ev["repository"] = repository(c, os.environ.get("GITHUB_TOKEN"))
    ev["sagemaker_quota"] = sagemaker(c, a.aws_profile)
    if not a.skip_tests:
        ev["tests"] = tests()
    if a.defence_video_url:
        q = urllib.parse.urlencode({"url": a.defence_video_url, "format": "json"})
        code, meta = call(f"https://www.youtube.com/oembed?{q}")
        ev["defence_video"] = {
            "url": a.defence_video_url,
            "title": meta["title"] if code == 200 else None,
            "checked_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),  # noqa: UP017
        }
    for item in MANUAL:
        c.rows.append({"group": "manual", "item": item, "status": "manual", "detail": ""})
    ev["checks"] = c.rows
    ev["collected_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017
    OUT.write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    failed = [r["item"] for r in c.rows if r["status"] == "fail"]
    print(f"\n{len(c.rows)} lines · failed: {failed or 'none'} -> {OUT.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
