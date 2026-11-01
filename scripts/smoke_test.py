"""Post-deploy smoke test (CD) — also runnable against a local container or the live URL.

    python scripts/smoke_test.py --api https://<api-url> --ui https://<ui-url> --expect-sha <sha>

Asserts: /health, /ready (retried while the service warms up), /version reports the expected commit,
the golden prediction is reproduced by the deployed model and routed by the campaign policy (D7),
/campaign returns a list, /recommend fires a rule, /chat answers a policy question with a citation,
and the interface answers its health probe. Standard library only, so it runs on a bare CI runner.
Exit code 1 on the first failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def call(url: str, body: dict[str, Any] | None = None, timeout: float = 30) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(  # noqa: S310 - URL comes from the operator
        url, data=data, headers={"Content-Type": "application/json", "X-Traffic-Cohort": "smoke"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            raw = r.read()
            return r.status, json.loads(raw) if raw[:1] in (b"{", b"[") else raw.decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--ui", default=None)
    ap.add_argument("--expect-sha", default=None)
    ap.add_argument("--wait-seconds", type=int, default=300)
    a = ap.parse_args()
    api = a.api.rstrip("/")
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        if not ok:
            sys.exit(1)

    def wait_for(url: str) -> tuple[int, Any]:
        deadline = time.time() + a.wait_seconds
        while True:
            try:
                code, body = call(url, timeout=10)
            except OSError as exc:  # refused / reset while the service starts
                code, body = 0, str(exc)
            if code == 200 or time.time() > deadline:
                return code, body
            time.sleep(10)

    code, body = wait_for(f"{api}/ready")
    check("ready", code == 200, json.dumps(body)[:200])
    code, body = call(f"{api}/health")
    check("health", code == 200 and body == {"status": "ok"}, str(body))

    code, v = call(f"{api}/version")
    sha_ok = a.expect_sha is None or v.get("git_commit") == a.expect_sha
    detail = f"git_commit={v.get('git_commit')} expected={a.expect_sha} tag={v.get('image_tag')}"
    check("version matches commit", code == 200 and sha_ok, detail)

    golden = json.loads((ROOT / "tests" / "golden" / "golden_predictions.json").read_text("utf-8"))
    case = golden["cases"][0]
    code, p = call(f"{api}/predict", {"customer_ref": case["ref"], "features": case["features"]})
    ok = code == 200 and abs(p["p_return"] - case["p_return"]) < 1e-6
    check("golden prediction", ok, f"p_return={p.get('p_return') if code == 200 else p}")

    code, cp = call(f"{api}/campaign")
    ok = code == 200 and sum(cp["counts"].values()) > 0 and p.get("action") is not None
    check("campaign policy (review list, hold-out)", ok, str(cp["counts"] if ok else cp)[:120])

    code, r = call(f"{api}/recommend", {"basket": ["22916", "22918"], "k": 3})
    check("recommend", code == 200 and r["strategy"].startswith("rules"), r.get("strategy", r))

    code, c = call(f"{api}/chat", {"message": "How long do refunds take?", "provider": "rule"})
    ok = code == 200 and c["tool"] == "policy_search" and bool(c["citations"])
    check(
        "chat (rule provider)", ok, f"tool={c.get('tool')} citations={len(c.get('citations', []))}"
    )

    if a.ui:
        code, h = wait_for(f"{a.ui.rstrip('/')}/_stcore/health")
        check("interface health", code == 200, str(h)[:60])
    print(f"\nsmoke test passed: {len(results)} checks against {api}")


if __name__ == "__main__":
    main()
