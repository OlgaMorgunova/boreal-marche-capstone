"""D5 scalability probe: where the system breaks as traffic, customers, rules or documents grow.

    python scripts/scalability_probe.py            # all probes, ~3 min
    python scripts/scalability_probe.py --only rules retrieval

Each probe grows one dimension and measures the cost, so the report's breaking points are
extrapolated from measured slopes, not guessed:

* ``throughput`` — one uvicorn worker (the container's CMD), N concurrent clients for a fixed time:
  requests/s and p95 per endpoint (where a single instance saturates);
* ``scoring`` — batch scoring time for 1k..100k customers (the quarterly campaign run);
* ``rules`` — the rule engine with 1x..64x the mined rule count: load time, memory, /recommend cost;
* ``retrieval`` — brute-force cosine search over 10^2..10^6 chunk vectors (dim 384);
* ``logstore`` — SQLite bytes per logged prediction and the monitoring report's cost as its
  window grows.

Writes ``reports/metrics/scalability.json``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latency_profile import free_port, start_server, summary  # noqa: E402

from boreal_marche.config import Paths, load_params  # noqa: E402
from boreal_marche.recommend.engine import Rule, RuleEngine  # noqa: E402
from boreal_marche.utils import read_json, utc_now, write_json  # noqa: E402

OUT = ROOT / "reports" / "metrics" / "scalability.json"
P = Paths()


def timed(fn: Any, repeat: int = 1) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def probe_throughput(seconds: float, concurrency: list[int]) -> dict[str, Any]:
    demo = read_json(P.demo_customers)["customers"]
    batch = {"items": [{"features": c["features"]} for c in demo[:100]]}
    cases = {
        "/predict": ("POST", "/predict", lambda i: {"features": demo[i % len(demo)]["features"]}),
        "/predict/batch (100)": ("POST", "/predict/batch", lambda i: batch),
        "/recommend": ("POST", "/recommend", lambda i: {"basket": ["85123A", "22423"], "k": 5}),
        "/chat policy (rule)": (
            "POST",
            "/chat",
            lambda i: {"message": "What is the returns policy?"},
        ),
    }
    tmp = Path(tempfile.mkdtemp(prefix="boreal_scale_"))
    port = free_port()
    proc, _ = start_server(tmp / "scale.db", port, {"BOREAL_MODEL_SOURCE": "file"})
    out: dict[str, Any] = {"workers": 1, "seconds_per_cell": seconds, "cells": []}
    try:
        for name, (method, path, body) in cases.items():
            for c in concurrency:
                lat: list[float] = []
                deadline = time.perf_counter() + seconds

                def worker(
                    seed: int,
                    deadline: float = deadline,
                    method: str = method,
                    path: str = path,
                    body: Any = body,
                ) -> list[float]:
                    mine, i = [], seed
                    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as cl:
                        while time.perf_counter() < deadline:
                            t0 = time.perf_counter()
                            cl.request(method, path, json=body(i)).raise_for_status()
                            mine.append(1000 * (time.perf_counter() - t0))
                            i += 7
                    return mine

                t0 = time.perf_counter()
                with cf.ThreadPoolExecutor(c) as pool:
                    for part in pool.map(worker, range(c)):
                        lat += part
                wall = time.perf_counter() - t0
                cell = {
                    "endpoint": name,
                    "concurrency": c,
                    "requests": len(lat),
                    "rps": round(len(lat) / wall, 1),
                    **summary(lat),
                }
                out["cells"].append(cell)
                print(f"  {name:22s} c={c:2d} {cell['rps']:7.1f} req/s  p95 {cell['p95']:7.1f} ms")
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    return out


def probe_scoring(sizes: list[int]) -> dict[str, Any]:
    from boreal_marche.retention.cv import load_gold

    model = joblib.load(P.model_file)
    val = load_gold(P).query("split == 'validation'")[model.feature_names]
    rows = []
    for n in sizes:
        X = val.sample(n=n, replace=True, random_state=0).reset_index(drop=True)
        s = timed(lambda X=X: model.predict_proba(X), repeat=3)
        rows.append({"n": n, "seconds": round(s, 3), "rows_per_s": round(n / s)})
        print(f"  score {n:>7,} customers: {s:.2f} s")
    return {"model_version": model.version, "rows": rows}


def scaled_engine(base: dict[str, Any], factor: int) -> RuleEngine:
    """``factor`` copies of every mined rule, each copy with a distinct synthetic consequent: the
    antecedent index grows exactly as it would with ``factor`` times more rules."""
    rules = [
        Rule(
            k * 100_000 + r["id"],
            frozenset(r["antecedent"]),
            r["consequent"] if k == 0 else f"{r['consequent']}~{k}",
            r["support"],
            r["confidence"],
            r["lift"],
        )
        for k in range(factor)
        for r in base["rules"]
    ]
    return RuleEngine(rules, base["popularity"], base["catalogue"], min_lift=3.0)


def probe_rules(factors: list[int]) -> dict[str, Any]:
    base = read_json(P.rules_file)
    rng = random.Random(0)  # noqa: S311 - reproducible synthetic baskets, not security
    antecedents = [r["antecedent"] for r in base["rules"]]
    baskets = [
        sorted({*rng.choice(antecedents), *rng.sample(base["popularity"][:300], rng.randint(1, 6))})
        for _ in range(300)
    ]
    rows = []
    for f in factors:
        tracemalloc.start()
        t0 = time.perf_counter()
        eng = scaled_engine(base, f)
        build = time.perf_counter() - t0
        mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        lat, checked = [], []
        for b in baskets:
            t0 = time.perf_counter()
            res = eng.recommend(b, k=5)
            lat.append(1000 * (time.perf_counter() - t0))
            checked.append(res.rules_checked)
        rows.append(
            {
                "factor": f,
                "n_rules": len(eng.rules),
                "build_s": round(build, 3),
                "memory_mb": round(mem / 1e6, 1),
                "rules_checked_mean": round(statistics.fmean(checked)),
                **summary(lat),
            }
        )
        print(
            f"  rules x{f:<3d} {len(eng.rules):>8,}: p95 {rows[-1]['p95']:.2f} ms, "
            f"{rows[-1]['memory_mb']} MB, build {build:.2f} s"
        )
        del eng
    return {"mined_rules": len(base["rules"]), "rows": rows}


def probe_retrieval(sizes: list[int], dim: int = 384) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    q = rng.standard_normal(dim).astype(np.float32)
    rows = []
    for n in sizes:
        V = rng.standard_normal((n, dim), dtype=np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)

        def search(V: np.ndarray = V) -> None:  # what Retriever.search does: matmul + sort
            scores = V @ q
            np.argsort(-scores)[:4]

        s = timed(search, repeat=5)
        rows.append(
            {"n_chunks": n, "ms": round(1000 * s, 2), "memory_mb": round(V.nbytes / 1e6, 1)}
        )
        print(f"  retrieval {n:>9,} chunks: {1000 * s:8.2f} ms, {V.nbytes / 1e6:7.1f} MB")
        del V
    n_chunks = read_json(P.index_dir / "index_meta.json")["n_chunks"]
    return {"dim": dim, "current_chunks": n_chunks, "rows": rows}


def probe_logstore(windows: list[int]) -> dict[str, Any]:
    from boreal_marche.monitoring.drift import build_report
    from boreal_marche.monitoring.store import LogStore

    params = load_params()
    prof = read_json(P.reference_profile)
    demo = read_json(P.demo_customers)["customers"]
    db = Path(tempfile.mkdtemp(prefix="boreal_log_")) / "log.db"
    store = LogStore(f"sqlite:///{db.as_posix()}")
    n = max(windows)
    for i in range(n):
        c = demo[i % len(demo)]
        store.log(
            "prediction_log",
            {
                "request_id": f"{i:036d}",
                "customer_ref": c["ref"],
                "model_version": "probe",
                "p_return": 0.5,
                "send_coupon": True,
                "threshold": 0.6,
                "features": json.dumps(c["features"]),
                "cohort": None,
            },
        )
        store.log(
            "request_log",
            {
                "request_id": f"{i:036d}",
                "method": "POST",
                "path": "/predict",
                "status": 200,
                "latency_ms": 4.0,
                "caller": "external",
                "cohort": None,
            },
        )
    store.flush()
    size = db.stat().st_size
    rows = []
    for w in windows:
        m = params.monitoring.model_copy(update={"window_predictions": w, "window_requests": w})
        s = timed(lambda m=m: build_report(store, prof, m), repeat=3)
        rows.append({"window": w, "report_ms": round(1000 * s, 1)})
        print(f"  monitoring report over {w:>6,} predictions: {1000 * s:.0f} ms")
    store.close()
    return {
        "rows_logged": n,
        "db_bytes": size,
        "bytes_per_prediction_and_request": round(size / n),
        "report": rows,
    }


PROBES = ("throughput", "scoring", "rules", "retrieval", "logstore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=PROBES, default=list(PROBES))
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    doc = read_json(OUT) if OUT.exists() else {}
    doc["machine"] = f"{sys.platform}, {os.cpu_count()} logical CPUs"
    for name in a.only:
        print(f"== {name}")
        if name == "throughput":
            doc[name] = probe_throughput(a.seconds, [1, 4, 16])
        elif name == "scoring":
            doc[name] = probe_scoring([1_000, 10_000, 100_000])
        elif name == "rules":
            doc[name] = probe_rules([1, 4, 16, 64])
        elif name == "retrieval":
            doc[name] = probe_retrieval([100, 10_000, 100_000, 1_000_000])
        elif name == "logstore":
            doc[name] = probe_logstore([500, 2_000, 10_000, 50_000])
        doc[f"{name}_measured_at"] = utc_now()
        write_json(OUT, doc)
    print(json.dumps({k: v for k, v in doc.items() if k.endswith("_at")}, indent=1))


if __name__ == "__main__":
    pd.set_option("mode.copy_on_write", True)
    main()
