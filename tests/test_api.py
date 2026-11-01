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
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    os.environ["BOREAL_MODEL_SOURCE"] = "local"
    os.environ["BOREAL_EMBEDDER"] = "hash"  # no model download in CI; retrieval quality is
    os.environ["BOREAL_MONITOR_LOOP"] = "0"  # evaluated separately (assistant_eval.json)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('db') / 'log.db'}"
    from boreal_marche import api

    with TestClient(api.app) as c:
        api.STATE["assistant"].toolbox.api = c  # the assistant's tools call this same app
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


# ----------------------------------------------------------------------------------- D4 ---

ME = "BM-E9B033CE"


def test_ready_reports_every_component(client: TestClient) -> None:
    body = client.get("/ready").json()
    assert body["assistant_loaded"] and body["model_loaded"] and body["rules_loaded"]
    assert "image_tag" in client.get("/version").json()


def test_customers_and_products(client: TestClient) -> None:
    refs = client.get("/customers", params={"limit": 3}).json()["customers"]
    assert len(refs) == 3 and refs[0]["ref"].startswith("BM-")
    feats = client.get(f"/customers/{refs[0]['ref']}/features")
    assert feats.status_code == 200 and "recency_days" in feats.json()
    assert client.get("/customers/BM-00000000/features").status_code == 404
    prods = client.get("/products", params={"q": "lunch bags"}).json()["products"]
    assert prods and all("LUNCH BAG" in p["description"] for p in prods)
    assert client.get("/products", params={"q": "22423"}).json()["products"][0]["sku"] == "22423"


def test_chat_contract(client: TestClient) -> None:
    r = client.post("/chat", json={"message": "Do my loyalty points expire?", "customer_ref": ME})
    body = r.json()
    assert r.status_code == 200 and body["tool"] == "policy_search" and body["citations"]
    r = client.post("/chat", json={"message": "What is my retention score?", "customer_ref": ME})
    body = r.json()  # the tool went through /customers and /predict of this API
    assert body["tool_result"]["customer_ref"] == ME and 0 <= body["tool_result"]["p_return"] <= 1
    oos = client.post("/chat", json={"message": "Write me a poem.", "customer_ref": ME}).json()
    assert oos["refused"] and oos["refusal_reason"] == "out_of_scope"
    assert client.post("/chat", json={"message": "hi", "oops": 1}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "provider": "gpt9"}).status_code == 422
    assert client.post("/chat", json={"message": "hi", "customer_ref": "42"}).status_code == 422
    status = {p["name"]: p for p in client.get("/chat/providers").json()}
    if not status["openai"]["available"]:
        assert client.post("/chat", json={"message": "hi", "provider": "openai"}).status_code == 503


def test_chat_stream_sse(client: TestClient) -> None:
    with client.stream(
        "POST", "/chat/stream", json={"message": "How long do refunds take?", "customer_ref": ME}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [ln[7:] for ln in r.iter_lines() if ln.startswith("event: ")]
    assert events[0] == "meta" and events[-1] == "done" and "token" in events


def test_predictions_are_logged_and_monitored(client: TestClient) -> None:
    from boreal_marche import api

    store = api.STATE["store"]
    store.flush()
    before = store.count("prediction_log")
    customers = json.loads((REPO / "artifacts" / "demo_customers.json").read_text("utf-8"))
    items = [
        {"customer_ref": c["ref"], "features": c["features"]} for c in customers["customers"][:120]
    ]
    assert client.post("/predict/batch", json={"items": items}).status_code == 200
    store.flush()
    assert store.count("prediction_log") == before + 120
    report = client.get("/monitoring/report", params={"fresh": True}).json()
    checks = {c["name"]: c for c in report["checks"]}
    assert checks["input_drift"]["status"] != "insufficient_data"
    assert set(checks) >= {"prediction_drift", "data_quality", "error_rate", "rule_fire_rate"}
    assert isinstance(client.get("/monitoring/alerts").json(), list)
    assert isinstance(client.get("/monitoring/timeseries").json(), list)
    metrics = client.get("/metrics").text
    assert "boreal_requests_total" in metrics and "boreal_monitoring_status" in metrics
    assert client.get("/health").headers["X-Request-ID"]
