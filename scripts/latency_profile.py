"""D5 latency profile of the API, by endpoint: cold start, first call, warm percentiles.

    python scripts/latency_profile.py --label before            # local uvicorn, 3 cold starts
    python scripts/latency_profile.py --label after
    python scripts/latency_profile.py --label live --url https://<app-runner-url> -n 40
    python scripts/latency_profile.py --label container_d5 --image boreal-api:d5

Local mode starts ``uvicorn boreal_marche.api:app`` in a subprocess (monitor loop off, a fresh
SQLite log), measures spawn -> ``/ready`` = 200 (cold start), then the **first** call to every
endpoint (cold path: lazy initialisation in XGBoost, ONNX Runtime, pandas) and ``n`` sequential
warm calls over one keep-alive connection. Server-side latency is read back from ``request_log``
(the API middleware's own timer), so network/client time and handler time are separated.

It also times each artifact load on its own — the cost **every request** would pay if the model,
rules or index were loaded per request instead of once at startup.

Results are merged into ``reports/metrics/latency_profile.json`` under ``runs.<label>``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boreal_marche.utils import read_json, utc_now, write_json  # noqa: E402

OUT = ROOT / "reports" / "metrics" / "latency_profile.json"


def endpoints(api: httpx.Client) -> list[tuple[str, str, str, Any]]:
    """(label, method, path, json body). Features come from the profiled service's own
    ``/customers`` endpoints, so the bodies match whichever API version is deployed."""
    refs = [c["ref"] for c in api.get("/customers", params={"limit": 100}).json()["customers"]]
    demo = [{"ref": r, "features": api.get(f"/customers/{r}/features").json()} for r in refs]
    c0 = demo[0]
    batch = [{"customer_ref": c["ref"], "features": c["features"]} for c in demo]
    basket = ["85123A", "22423", "47566"]
    return [
        ("/health", "GET", "/health", None),
        ("/ready", "GET", "/ready", None),
        ("/version", "GET", "/version", None),
        ("/customers", "GET", "/customers?limit=50", None),
        ("/customers/{ref}/features", "GET", f"/customers/{c0['ref']}/features", None),
        ("/products", "GET", "/products?q=lantern&limit=10", None),
        ("/predict", "POST", "/predict", {"customer_ref": c0["ref"], "features": c0["features"]}),
        ("/predict/batch (100)", "POST", "/predict/batch", {"items": batch}),
        ("/recommend", "POST", "/recommend", {"basket": basket, "k": 5}),
        ("/chat policy (rule)", "POST", "/chat",
         {"message": "How long do I have to return a product?", "provider": "rule"}),
        ("/chat retention tool (rule)", "POST", "/chat",
         {"message": "Will I get a discount coupon this quarter?", "provider": "rule",
          "customer_ref": c0["ref"]}),
        ("/monitoring/report", "GET", "/monitoring/report", None),
    ]  # fmt: skip


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def summary(xs: list[float]) -> dict[str, float]:
    return {"n": len(xs), "p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "p99": pct(xs, 0.99),
            "mean": round(statistics.fmean(xs), 2)}  # fmt: skip


def call(api: httpx.Client, method: str, path: str, body: Any) -> float:
    t0 = time.perf_counter()
    r = api.request(method, path, json=body, headers={"X-Traffic-Cohort": "perf"})
    ms = 1000 * (time.perf_counter() - t0)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text[:200]}")
    return ms


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_server(db: Path, port: int, extra_env: dict[str, str]) -> tuple[subprocess.Popen, float]:
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1", BOREAL_MONITOR_LOOP="0",
        DATABASE_URL=f"sqlite:///{db.as_posix()}", BOREAL_API_URL=f"http://127.0.0.1:{port}",
        LOG_LEVEL="WARNING", **extra_env,
    )  # fmt: skip
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "boreal_marche.api:app", "--port", str(port),
         "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )  # fmt: skip
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/ready", timeout=2).status_code == 200:
                return proc, 1000 * (time.perf_counter() - t0)
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise RuntimeError(proc.stderr.read().decode()[-2000:] if proc.stderr else "exited")
        time.sleep(0.05)
    proc.kill()
    raise TimeoutError("server not ready after 180 s")


def server_side(db: Path) -> dict[str, dict[str, float]]:
    """Middleware latency per route template, warm calls only (the first call is skipped)."""
    con = sqlite3.connect(db)
    rows = con.execute(  # external calls only: the assistant's own /predict calls are internal
        "SELECT path, latency_ms FROM request_log WHERE caller = 'external' ORDER BY id"
    ).fetchall()
    con.close()
    by: dict[str, list[float]] = {}
    for path, ms in rows:
        by.setdefault(path, []).append(ms)
    return {p: summary(v[1:]) for p, v in by.items() if len(v) > 2}


def artifact_loads(repeat: int = 3) -> dict[str, float]:
    """Median time (ms) to load each artifact once — the per-request cost of lazy loading."""
    code = r"""
import json, time, joblib
from boreal_marche.config import Paths, load_params
from boreal_marche.recommend.engine import RuleEngine
from boreal_marche.assistant.index import load_retriever, make_embedder
from boreal_marche.utils import read_json
p, P = load_params(), Paths()
def t(fn):
    t0 = time.perf_counter(); fn(); return 1000 * (time.perf_counter() - t0)
rules = lambda: RuleEngine.from_json(P.rules_file, min_lift=p.recommend.min_lift)
out = {
  "model.joblib": t(lambda: joblib.load(P.model_file)),
  "rules.json -> RuleEngine": t(rules),
  "embedding model (ONNX)": t(lambda: make_embedder(p)),
  "retrieval index": 0.0,
  "reference_profile.json": t(lambda: read_json(P.reference_profile)),
}
emb = make_embedder(p)
out["retrieval index"] = t(lambda: load_retriever(p, P, emb))
# Libraries are imported now: what a per-request load would cost on a warm process.
out["reload: model.joblib"] = t(lambda: joblib.load(P.model_file))
out["reload: rules.json -> RuleEngine"] = t(rules)
out["reload: embedding model (ONNX)"] = t(lambda: make_embedder(p))
print(json.dumps(out))
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONUTF8="1")
    runs = []
    for _ in range(repeat):
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                           text=True, check=True)  # fmt: skip
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return {k: round(statistics.median(r[k] for r in runs), 1) for k in runs[0]}


def profile_local(n: int, cold_starts: int, extra_env: dict[str, str]) -> dict[str, Any]:
    starts, first_calls, warm, server, startup = [], {}, {}, {}, {}
    for i in range(cold_starts):
        tmp = Path(tempfile.mkdtemp(prefix="boreal_perf_"))
        db, port = tmp / "perf.db", free_port()
        proc, ready_ms = start_server(db, port, extra_env)
        starts.append(ready_ms)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as api:
                eps = endpoints(api)
                for label, method, path, body in eps:
                    first_calls.setdefault(label, []).append(call(api, method, path, body))
                if i == cold_starts - 1:  # warm percentiles on the last server only
                    for label, method, path, body in eps:
                        reps = max(5, n // 10) if "batch" in label or "report" in label else n
                        warm[label] = summary([call(api, method, path, body) for _ in range(reps)])
                startup = api.get("/ready").json().get("startup_ms", {})
            time.sleep(1.5)  # let the log writer thread flush
            if i == cold_starts - 1:
                server = server_side(db)
        finally:
            proc.terminate()
            proc.wait(timeout=30)
    return {
        "cold_start_to_ready_ms": {"runs": [round(s) for s in starts],
                                   "median": round(statistics.median(starts))},
        "first_call_ms": {k: round(statistics.median(v), 1) for k, v in first_calls.items()},
        "warm_client_ms": warm,
        "warm_server_ms": server,
        "startup_ms_reported_by_ready": startup,
        "artifact_load_ms": artifact_loads(),
    }  # fmt: skip


def profile_container(image: str, cold_starts: int) -> dict[str, Any]:
    """`docker run` -> /ready = 200, i.e. the cold start App Runner pays (image already pulled),
    plus idle memory from `docker stats` and the startup timings the image reports."""
    starts, mem, startup = [], [], {}
    for _ in range(cold_starts):
        port = free_port()
        t0 = time.perf_counter()
        cid = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", f"{port}:8000", "-e", "BOREAL_MONITOR_LOOP=0",
             image], check=True, capture_output=True, text=True,
        ).stdout.strip()  # fmt: skip
        try:
            while time.perf_counter() - t0 < 180:
                try:
                    r = httpx.get(f"http://127.0.0.1:{port}/ready", timeout=2)
                    if r.status_code == 200:
                        starts.append(1000 * (time.perf_counter() - t0))
                        startup = r.json().get("startup_ms", {})
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            time.sleep(3)
            stats = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", cid],
                check=True, capture_output=True, text=True,
            ).stdout  # fmt: skip
            mem.append(stats.split("/")[0].strip())
        finally:
            subprocess.run(["docker", "stop", cid], capture_output=True, check=False)
    return {
        "image": image,
        "cold_start_to_ready_ms": {"runs": [round(s) for s in starts],
                                   "median": round(statistics.median(starts))},
        "idle_memory": mem,
        "startup_ms_reported_by_ready": startup,
    }  # fmt: skip


def profile_remote(url: str, n: int) -> dict[str, Any]:
    warm, first = {}, {}
    with httpx.Client(base_url=url.rstrip("/"), timeout=60) as api:
        for label, method, path, body in endpoints(api):
            if "batch" in label or "report" in label:
                continue  # keep load on the shared live instance small
            first[label] = round(call(api, method, path, body), 1)
            warm[label] = summary([call(api, method, path, body) for _ in range(n)])
    return {"url": url, "first_call_ms": first, "warm_client_ms": warm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="before | after | live | any name")
    ap.add_argument("--url", default=None, help="profile a deployed service instead")
    ap.add_argument("--image", default=None, help="container cold start of a local image")
    ap.add_argument("-n", type=int, default=200, help="warm calls per endpoint")
    ap.add_argument("--cold-starts", type=int, default=3)
    ap.add_argument("--note", default="")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server")
    a = ap.parse_args()
    extra = dict(e.split("=", 1) for e in a.env)
    if a.image:
        res = profile_container(a.image, a.cold_starts)
    elif a.url:
        res = profile_remote(a.url, a.n)
    else:
        res = profile_local(a.n, a.cold_starts, extra)
    res |= {"measured_at": utc_now(), "note": a.note, "server_env": extra,
            "machine": f"{sys.platform}, Python {sys.version.split()[0]}, {os.cpu_count()} logical CPUs"}  # fmt: skip  # noqa: E501
    doc = read_json(OUT) if OUT.exists() else {"runs": {}}
    doc["runs"][a.label] = res
    write_json(OUT, doc)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
