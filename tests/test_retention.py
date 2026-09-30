"""Head A building blocks: baselines, metrics, calibration, profit, artifact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boreal_marche.config import BusinessParams
from boreal_marche.features import CATEGORICAL, FEATURES
from boreal_marche.retention.artifact import Calibrator, RetentionModel
from boreal_marche.retention.models import (
    RecencyDeciles,
    RecencyRule,
    build,
    model_zoo,
    recall_at_top,
    score_all,
)
from boreal_marche.retention.profit import (
    best_threshold,
    expected_profit,
    profit_curve,
    realised_profit,
)

B = BusinessParams(
    coupon_cost=4.0,
    discount_rate=0.15,
    save_rate=0.2,
    gross_margin=0.3,
    value_horizon_days=365,
    sensitivity_save_rates=[0.1, 0.2],
)


def synthetic_features(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({c: rng.gamma(2.0, 10.0, n) for c in FEATURES if c not in CATEGORICAL})
    X["region"] = rng.choice(["QC", "ON"], n)
    X["season_q"] = rng.choice(["1", "3"], n)
    X.loc[X.sample(frac=0.2, random_state=0).index, "mean_gap_days"] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp((X["recency_days"] - 20) / 10))).astype(int)
    return X[FEATURES], y.to_numpy()


def test_at_least_eight_families() -> None:
    assert len(model_zoo(0)) >= 8


def test_recency_rule_and_deciles(rng) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_features(400, rng)
    rule = RecencyRule(20).fit(X, y)
    p = rule.predict_proba(X)[:, 1]
    assert set(np.round(p, 6)) == {round(rule.rate_recent_, 6), round(rule.rate_other_, 6)}
    dec = RecencyDeciles().fit(X, y)
    assert dec.predict_proba(X).shape == (400, 2)


def test_pipeline_handles_nan_and_categories(rng) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_features(300, rng)
    pipe = build("logreg", 0).fit(X, y)
    assert np.isfinite(pipe.predict_proba(X)).all()


def test_metrics() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.linspace(1, 0, 10)
    m = score_all(y, p)
    assert 0 <= m["pr_auc"] <= 1 and 0 <= m["brier"] <= 1
    assert recall_at_top(y, p, 0.1) == pytest.approx(0.2)


@pytest.mark.parametrize("method", ["none", "sigmoid", "isotonic"])
def test_calibrator(method: str, rng) -> None:  # type: ignore[no-untyped-def]
    p = rng.random(500)
    y = (rng.random(500) < p**2).astype(int)
    out = Calibrator(method).fit(p, y).transform(p)
    assert ((out >= 0) & (out <= 1)).all()


def test_profit_math() -> None:
    y = np.array([0, 1])
    value = np.array([100.0, 100.0])
    disc = np.array([10.0, 10.0])
    real = realised_profit(y, value, disc, B)
    assert real[0] == pytest.approx(0.2 * 90 - 4)
    assert real[1] == pytest.approx(-14)
    exp = expected_profit(np.array([0.0, 1.0]), value, disc, B)
    np.testing.assert_allclose(exp, real)


def test_threshold_prefers_lapsers() -> None:
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    v = np.full(4, 1000.0)
    d = np.full(4, 10.0)
    c = profit_curve(p, y, v, d, B)
    t = best_threshold(c)
    assert 0.2 <= t < 0.8
    assert c.loc[c["threshold"] == 1.0, "n_contacted"].item() == 4


def test_artifact_contract(rng) -> None:  # type: ignore[no-untyped-def]
    X, y = synthetic_features(300, rng)
    pipe = build("logreg", 0).fit(X, y)
    m = RetentionModel(pipe, Calibrator("none"), list(FEATURES), 0.5, "test")
    assert m.predict_proba(X.sample(frac=1, axis=1)).shape == (300, 2)  # column order irrelevant
    assert m.decide(X).dtype == bool
    with pytest.raises(ValueError):
        m.predict_proba(X.drop(columns=["recency_days"]))
