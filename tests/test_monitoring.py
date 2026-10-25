"""Monitoring tests: PSI/KS maths, traffic lights, the log store, the alert state machine, and an
end-to-end report on normal versus deliberately shifted traffic built from the committed reference.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boreal_marche.config import Band, load_params
from boreal_marche.monitoring import alerts
from boreal_marche.monitoring.drift import band, build_report, feature_drift, psi, worst
from boreal_marche.monitoring.store import LogStore

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "artifacts" / "reference_profile.json"
DEMO = REPO / "artifacts" / "demo_customers.json"


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[LogStore]:
    s = LogStore(f"sqlite:///{tmp_path / 'log.db'}")
    yield s
    s.close()


def test_psi_properties() -> None:
    e = np.array([0.25, 0.25, 0.25, 0.25])
    assert psi(e, e) == pytest.approx(0.0, abs=1e-9)
    assert psi(e, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25
    assert psi(e, np.array([0.3, 0.25, 0.25, 0.2])) < 0.1


def test_bands_and_worst() -> None:
    b = Band(amber=0.1, red=0.25)
    assert [band(v, b) for v in (0.05, 0.1, 0.3)] == ["green", "amber", "red"]
    assert band(0.2, Band(amber=0.5, red=0.3), lower_is_worse=True) == "red"
    assert band(None, b) == "insufficient_data"
    assert worst(["green", "insufficient_data", "amber"]) == "amber"
    assert worst(["insufficient_data"]) == "insufficient_data"


def test_feature_drift_numeric_and_categorical() -> None:
    m = load_params().monitoring
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(size=5000)
    edges = np.quantile(ref_vals, np.linspace(0.1, 0.9, 9)).tolist()
    ref = {
        "type": "numeric",
        "bin_edges": edges,
        "proportions": [0.1] * 10,
        "missing_rate": 0.0,
        "mean": 0.0,
        "sample": ref_vals[:2000].tolist(),
    }
    same = feature_drift("x", ref, pd.Series(rng.normal(size=500)), m)
    shifted = feature_drift("x", ref, pd.Series(rng.normal(loc=1.5, size=500)), m)
    assert same["status"] == "green" and shifted["status"] == "red" and shifted["ks"] > 0.2
    cat = {"type": "categorical", "categories": {"QC": 0.9, "ON": 0.1}, "missing_rate": 0.0}
    moved = feature_drift("region", cat, pd.Series(["MARITIMES"] * 300 + ["QC"] * 200), m)
    assert moved["psi"] > 0.25 and moved["ks"] is None


def test_store_roundtrip_and_alert_transitions(store: LogStore) -> None:
    for i in range(5):
        store.log("request_log", {"path": "/predict", "status": 200, "latency_ms": float(i)})
    store.flush()
    assert store.count("request_log") == 5
    assert list(store.recent("request_log", 2)["latency_ms"]) == [3.0, 4.0]

    def report(level: str) -> dict:
        return {
            "checks": [
                {"name": "error_rate", "status": level, "value": 0.1, "unit": "u", "threshold": {}}
            ]
        }

    assert len(alerts.evaluate(store, report("red"))) == 1
    assert alerts.evaluate(store, report("red")) == []  # sustained breach notifies once
    raised = alerts.evaluate(store, report("green"))
    assert raised[0]["previous"] == "red" and raised[0]["level"] == "green"
    assert store.last_levels() == {"error_rate": "green"}


@pytest.mark.skipif(not (PROFILE.exists() and DEMO.exists()), reason="reference not built")
def test_report_turns_red_on_shifted_traffic(store: LogStore) -> None:
    params = load_params()
    profile = json.loads(PROFILE.read_text("utf-8"))
    customers = json.loads(DEMO.read_text("utf-8"))["customers"]

    def log(feats: dict, cohort: str) -> None:
        store.log(
            "prediction_log",
            {
                "customer_ref": "x",
                "p_return": 0.5,
                "send_coupon": True,
                "features": json.dumps(feats),
                "cohort": cohort,
            },
        )
        store.log("request_log", {"path": "/predict", "status": 200, "latency_ms": 5.0})

    for c in customers[:200]:
        log(c["features"], "normal")
    store.flush()
    normal = build_report(store, profile, params.monitoring)
    drift = next(c for c in normal["checks"] if c["name"] == "input_drift")
    assert drift["status"] in ("green", "amber") and normal["window"]["predictions"] == 200

    for c in customers[:500]:
        f = dict(c["features"])
        f.update(
            recency_days=f["recency_days"] * 4 + 60,
            revenue_180d=f["revenue_180d"] * 0.2,
            region="MARITIMES",
            mean_gap_days=None,
            std_gap_days=None,
            overdue_ratio=None,
        )
        log(f, "shifted")
    store.flush()
    shifted = build_report(store, profile, params.monitoring)
    status = {c["name"]: c["status"] for c in shifted["checks"]}
    assert status["input_drift"] == "red" and status["data_quality"] == "red"
    assert shifted["status"] == "red"
    assert {"recency_days", "region"} <= {f["feature"] for f in shifted["features"][:6]}
