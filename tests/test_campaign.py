"""D7: the served fairness mitigation — high-value review list and randomised hold-out
(``retention/campaign.py``), unit-level and through the API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from boreal_marche.retention.campaign import CampaignPolicy, holdout_power

REPO = Path(__file__).resolve().parents[1]
POLICY_FILE = REPO / "artifacts" / "campaign_policy.json"
GOLDEN = json.loads((REPO / "tests" / "golden" / "golden_predictions.json").read_text("utf-8"))
needs_artifacts = pytest.mark.skipif(not POLICY_FILE.exists(), reason="artifacts not built")

POLICY = CampaignPolicy(
    campaign_id="test",
    model_version="m",
    threshold=0.6,
    spend_edges=[100.0, 200.0, 300.0],
    review_thresholds={"Q3": 0.7, "Q4 (top)": 0.9},
    holdout_share=0.1,
    source="test",
)


def test_spend_quartile_matches_right_closed_bins() -> None:
    assert [POLICY.spend_quartile(x) for x in (50, 100, 100.01, 200, 250, 300, 301)] == [
        "Q1 (low)", "Q1 (low)", "Q2", "Q2", "Q3", "Q3", "Q4 (top)",
    ]  # fmt: skip


def test_actions() -> None:
    ref = next(f"r{i}" for i in range(1000) if not POLICY.in_holdout(f"r{i}"))
    assert POLICY.action(0.5, 50, ref) == "coupon"
    assert POLICY.action(0.65, 50, ref) == "none"  # low spender: no review list
    assert POLICY.action(0.65, 250, ref) == "review"  # Q3, under its group threshold
    assert POLICY.action(0.75, 250, ref) == "none"  # Q3, above it
    assert POLICY.action(0.85, 400, ref) == "review"  # Q4
    assert POLICY.action(0.95, 400, ref) == "none"


def test_holdout_is_deterministic_feature_blind_and_sized() -> None:
    refs = [f"BM-{i:08X}" for i in range(20_000)]
    held = [POLICY.in_holdout(r) for r in refs]
    assert held == [POLICY.in_holdout(r) for r in refs]
    assert 0.09 < sum(held) / len(held) < 0.11
    assert not POLICY.in_holdout(None)
    held_out = next(r for r, h in zip(refs, held, strict=True) if h)
    assert POLICY.action(0.1, 50, held_out) == "holdout"  # only proposed customers are held out
    assert POLICY.action(0.65, 250, held_out) == "review"
    other = CampaignPolicy(**{**POLICY.__dict__, "campaign_id": "next-quarter"})
    assert [other.in_holdout(r) for r in refs[:2000]] != held[:2000]  # a new draw each campaign


def test_holdout_power_shrinks_with_size() -> None:
    small = holdout_power(1000, 100, 0.35, 0.65)
    assert holdout_power(4000, 400, 0.35, 0.65) == pytest.approx(small / 2, rel=0.01)
    assert holdout_power(1000, 0, 0.35, 0.65) == float("inf")


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    os.environ["BOREAL_MODEL_SOURCE"] = "local"
    os.environ["BOREAL_EMBEDDER"] = "hash"
    os.environ["BOREAL_MONITOR_LOOP"] = "0"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path_factory.mktemp('d7') / 'log.db'}"
    from boreal_marche import api

    with TestClient(api.app) as c:
        yield c


@needs_artifacts
def test_policy_artifact_matches_served_model(client: TestClient) -> None:
    policy = CampaignPolicy.load(POLICY_FILE)
    assert policy.model_version == client.get("/version").json()["model_version"]
    assert client.get("/ready").json()["errors"] == []


@needs_artifacts
def test_predict_carries_action_and_keeps_the_decision(client: TestClient) -> None:
    for case in GOLDEN["cases"]:
        body = client.post(
            "/predict", json={"customer_ref": case["ref"], "features": case["features"]}
        ).json()
        assert body["send_coupon"] == (body["p_return"] <= body["threshold"])
        assert body["action"] in (
            ("coupon", "holdout") if body["send_coupon"] else ("review", "none")
        )
        assert body["spend_quartile"]


@needs_artifacts
def test_campaign_list(client: TestClient) -> None:
    r = client.get("/campaign")
    assert r.status_code == 200
    c = r.json()
    assert sum(c["counts"].values()) == len(client.get("/customers?limit=300").json()["customers"])
    assert all(e["spend_quartile"] in c["review_thresholds"] for e in c["review_list"])
    assert all(e["p_return"] > c["threshold"] for e in c["review_list"])
    assert all(e["p_return"] <= c["threshold"] for e in c["holdout"])
    assert len(c["holdout"]) == c["counts"]["holdout"]
