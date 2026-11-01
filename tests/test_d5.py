"""D5: latency optimisations keep behaviour, exact per-prediction factors, data minimisation in the
API contract and log, cached monitoring report, fairness audit and the minimisation promotion gate.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from boreal_marche.features import COLLECTED, EXCLUDED_FROM_MODEL, FEATURES, MODEL_FEATURES
from boreal_marche.retention import fairness
from boreal_marche.retention.promote import decide

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "artifacts" / "model.joblib"
GOLDEN = json.loads((REPO / "tests" / "golden" / "golden_predictions.json").read_text("utf-8"))
needs_artifacts = pytest.mark.skipif(not MODEL.exists(), reason="artifacts not built")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    os.environ["BOREAL_MODEL_SOURCE"] = "local"
    os.environ["BOREAL_EMBEDDER"] = "hash"
    os.environ["BOREAL_MONITOR_LOOP"] = "0"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('d5') / 'log.db'}"
    from boreal_marche import api

    with TestClient(api.app) as c:
        yield c


def test_feature_sets_are_consistent() -> None:
    assert len(MODEL_FEATURES) == 19 and len(FEATURES) == 37
    assert set(MODEL_FEATURES) | set(EXCLUDED_FROM_MODEL) == set(FEATURES)
    assert "region" in COLLECTED and "region" not in MODEL_FEATURES
    assert "weekend_share_180d" not in COLLECTED and "mean_order_hour_180d" not in COLLECTED


@needs_artifacts
def test_fast_frame_equals_pandas_frame() -> None:
    from boreal_marche.api import _frame

    model = joblib.load(MODEL)
    rows = [c["features"] for c in GOLDEN["cases"]]
    ref = pd.DataFrame(rows)[model.feature_names]
    ref = ref.astype({c: "float64" for c in ref.columns if c not in ("region", "season_q")})
    fast = _frame(rows, model.feature_names)
    pd.testing.assert_frame_equal(fast, ref, check_dtype=True)
    assert np.array_equal(model.predict_proba(fast), model.predict_proba(ref))


@needs_artifacts
def test_contributions_are_exact_in_calibrated_log_odds() -> None:
    from boreal_marche.retention.explain import contributions, top_factors

    model = joblib.load(MODEL)
    X = pd.DataFrame([c["features"] for c in GOLDEN["cases"]])
    contrib, base = contributions(model, X)
    p = model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(contrib.sum(axis=1) + base, np.log(p / (1 - p)), atol=1e-4)
    for factors in top_factors(model, X, 3):
        assert len(factors) == 3 and all(f["feature"] != "season_q" for f in factors)
        effects = [abs(f["effect"]) for f in factors]
        assert effects == sorted(effects, reverse=True)


@needs_artifacts
def test_predict_explain_and_minimised_logging(client: TestClient) -> None:
    from boreal_marche import api

    full = GOLDEN["cases"][0]["features"]
    minimal = {k: v for k, v in full.items() if k in COLLECTED}
    a = client.post("/predict", json={"features": minimal}, params={"explain": 3}).json()
    b = client.post("/predict", json={"features": full | {"weekend_share_180d": 0.5}}).json()
    assert a["p_return"] == pytest.approx(b["p_return"])  # removed features have no effect
    assert len(a["top_factors"]) == 3 and b["top_factors"] is None
    assert {"feature", "label", "value", "effect", "direction"} <= set(a["top_factors"][0])
    store = api.STATE["store"]
    store.flush()
    logged = json.loads(store.recent("prediction_log", 1)["features"].iloc[0])
    assert set(logged) == set(COLLECTED)  # nothing outside the model inputs + region is stored


def test_ready_reports_startup_timings(client: TestClient) -> None:
    ms = client.get("/ready").json()["startup_ms"]
    assert {"model", "assistant", "engine", "warm_up", "total"} <= set(ms)


def test_monitoring_report_is_cached_unless_fresh(client: TestClient) -> None:
    first = client.get("/monitoring/report").json()["generated_at"]
    assert client.get("/monitoring/report").json()["generated_at"] == first
    assert client.get("/monitoring/report", params={"fresh": True}).json()["generated_at"] > first


def test_products_search_uses_folded_catalogue(client: TestClient) -> None:
    hits = client.get("/products", params={"q": "LANTERN"}).json()["products"]
    assert hits and all("lantern" in p["description"].lower() for p in hits)
    assert all("_folded" not in p for p in hits)  # internal field never leaks


def test_fairness_audit_flags_a_gap() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({
        "region": rng.choice(["QC", "ON"], n),
        "tenure_days": rng.uniform(0, 700, n),
        "revenue_180d": rng.gamma(2.0, 300.0, n),
        "y": rng.integers(0, 2, n),
    })  # fmt: skip
    edges = fairness.spend_edges(df)
    top = df["revenue_180d"].to_numpy() > edges[-1]
    send = (df["y"].to_numpy() == 0) & ~top  # every lapser contacted except the top spenders
    a = fairness.audit(df, rng.uniform(size=n), send, edges)
    assert a["overall"]["lapser_recall"] == pytest.approx(0.75, abs=0.03)
    assert "spend_quartile=Q4 (top)" in a["flagged"] and a["worst_gap"] > 0.5
    th = fairness.group_thresholds(df, rng.uniform(size=n), 0.5, edges)
    assert set(th) == set(fairness.SPEND_LABELS) and min(th.values()) >= 0.5


def test_promotion_gate_minimisation_clause() -> None:
    fair = {"worst_gap": 0.4}
    champ = {"pr_auc": 0.810, "n_features": 37, "fairness": fair}
    small = {"pr_auc": 0.806, "n_features": 19, "fairness": fair}
    assert decide(small, champ, {})["promote"]  # fewer features, loses <= 0.005
    worse = {"pr_auc": 0.800, "n_features": 19, "fairness": fair}
    assert not decide(worse, champ, {})["promote"]
    same_size = {"pr_auc": 0.806, "n_features": 37, "fairness": fair}
    assert not decide(same_size, champ, {})["promote"]  # no gain, no minimisation
