"""API contract tests, including the golden-prediction regression test.

The contract tests run against the committed artifacts in ``artifacts/`` (model.joblib, rules.json);
``BOREAL_MODEL_SOURCE=local`` keeps them independent of an MLflow server.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "golden" / "golden_predictions.json"
HAS_ARTIFACTS = (REPO / "artifacts" / "model.joblib").exists() and (
    REPO / "artifacts" / "rules.json"
).exists()
pytestmark = pytest.mark.skipif(not HAS_ARTIFACTS, reason="artifacts not built")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    os.environ["BOREAL_MODEL_SOURCE"] = "local"
    from boreal_marche.api import app

    with TestClient(app) as c:
        yield c


def golden() -> list[dict]:  # type: ignore[type-arg]
    return json.loads(GOLDEN.read_text())["cases"]


def test_health_and_ready(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["ready"] is True


def test_version(client: TestClient) -> None:
    v = client.get("/version").json()
    assert v["model_version"] and v["rule_count"] > 0 and v["git_commit"]


def test_predict_schema_and_extra_fields(client: TestClient) -> None:
    feats = golden()[0]["features"]
    r = client.post("/predict", json={"customer_ref": "c1", "features": feats})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["p_return"] <= 1 and body["customer_ref"] == "c1"
    assert body["send_coupon"] == (body["p_return"] <= body["threshold"])
    assert client.post("/predict", json={"features": feats, "oops": 1}).status_code == 422
    bad = dict(feats, region="MARS")
    assert client.post("/predict", json={"features": bad}).status_code == 422
    missing = {k: v for k, v in feats.items() if k != "recency_days"}
    assert client.post("/predict", json={"features": missing}).status_code == 422


def test_golden_predictions(client: TestClient) -> None:
    """Regression guard: the served model must reproduce the recorded probabilities."""
    cases = golden()
    r = client.post(
        "/predict/batch",
        json={"items": [{"customer_ref": c["ref"], "features": c["features"]} for c in cases]},
    )
    assert r.status_code == 200
    for case, pred in zip(cases, r.json()["predictions"], strict=True):
        assert pred["p_return"] == pytest.approx(case["p_return"], abs=1e-6), case["ref"]


def test_batch_limits(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"items": []}).status_code == 422


def test_recommend_strategies(client: TestClient) -> None:
    assert client.post("/recommend", json={"basket": []}).json()["strategy"] == "empty_basket"
    r = client.post("/recommend", json={"basket": ["NOT-A-SKU"], "k": 3}).json()
    assert r["strategy"] == "popularity_fallback" and r["unknown_items"] == ["NOT-A-SKU"]
    r = client.post("/recommend", json={"basket": ["22916", "22918"], "k": 3}).json()
    assert r["strategy"].startswith("rules")
    first = r["recommendations"][0]
    assert first["source"] == "rule" and first["because"] and first["lift"] >= 3.0
    assert client.post("/recommend", json={"basket": [], "k": 0}).status_code == 422
