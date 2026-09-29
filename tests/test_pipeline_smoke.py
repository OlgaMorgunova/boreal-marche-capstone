"""End-to-end smoke test: silver -> gold -> spot-check -> tune -> train -> test -> rules.

Runs on synthetic data in a temporary repository root, proving the pipeline works from a clean state
(CI has no access to the real dataset)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boreal_marche.config import Paths, load_params
from tests.conftest import make_lines

BUNDLES = [["A1", "A2", "A3"], ["B1", "B2"], ["C1", "C2", "C3", "C4"], ["D1"], ["E1", "E2"]]


def synthetic_silver(rng: np.random.Generator, n_customers: int = 240) -> pd.DataFrame:
    rows = []
    inv = 100000
    start, end = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")
    span = (end - start).days
    for c in range(n_customers):
        rate = rng.gamma(1.5, 1 / 40)  # orders per day
        t = float(rng.uniform(0, 120))
        while t < span:
            inv += 1
            day = start + pd.Timedelta(days=t, hours=int(rng.integers(8, 18)))
            for b in rng.choice(len(BUNDLES), size=int(rng.integers(1, 3)), replace=False):
                for item in BUNDLES[b]:
                    if rng.random() < 0.85:
                        rows.append(
                            (
                                str(inv),
                                c + 1,
                                item,
                                str(day),
                                int(rng.integers(1, 12)),
                                float(rng.choice([1.25, 2.1, 4.95])),
                            )
                        )
            t += float(rng.exponential(1 / rate)) + 1
    return make_lines(rows)


@pytest.fixture(scope="module")
def tmp_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Paths, object]:
    root = tmp_path_factory.mktemp("repo")
    paths = Paths(root)
    paths.ensure()
    base = load_params()
    params = base.model_copy(
        update={
            "tuning": base.tuning.model_copy(update={"n_trials": 3, "timeout_seconds": 120}),
            "recommend": base.recommend.model_copy(
                update={
                    "min_support_grid": [0.05, 0.1],
                    "min_confidence_grid": [0.3],
                    "min_support": 0.05,
                    "min_confidence": 0.3,
                    "min_lift": 1.5,
                }
            ),
            "mlflow": base.mlflow.model_copy(update={"tracking_uri": "sqlite:///mlflow.db"}),
        }
    )
    lines = synthetic_silver(np.random.default_rng(7))
    lines.to_parquet(paths.silver_lines, index=False)
    canc = lines.sample(frac=0.02, random_state=1)[["customer_id", "invoice_date", "stock_code"]]
    canc.to_parquet(paths.silver_cancellations, index=False)
    (paths.root / "data").mkdir(exist_ok=True)
    paths.manifest.write_text(json.dumps({"sha256": "synthetic"}))
    return paths, params


def test_full_pipeline(tmp_repo, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    paths, params = tmp_repo
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from boreal_marche import features
    from boreal_marche.recommend import evaluate as rec_eval
    from boreal_marche.recommend import mine
    from boreal_marche.retention import evaluate, leakage, spotcheck, train, tune

    gold = features.run(params, paths)
    assert set(gold["split"]) == {"train", "validation", "test"}
    spot = spotcheck.run(params, paths)
    assert len(spot["top2"]) == 2 and (paths.figures / "spotcheck_boxplot.png").exists()
    tuned = tune.run(params, paths)
    assert tuned["results"][tuned["winner"]]["n_trials"] == 3
    final = train.run(params, paths)
    assert paths.model_file.exists() and paths.model_meta.exists()
    assert 0.0 <= final["threshold"] <= 1.0
    res = evaluate.run(params, paths)
    assert res["n_test_looks"] == 1
    with pytest.raises(SystemExit):
        evaluate.run(params, paths)  # second look requires --force
    assert evaluate.run(params, paths, skip_if_done=True)["n_test_looks"] == 1
    meta = json.loads(paths.model_meta.read_text())
    assert meta["test_metrics"]["pr_auc"] == pytest.approx(res["metrics"]["pr_auc"])
    leak = leakage.run(params, paths)
    assert (
        leak["label_window_leak"]["leaky"]["pr_auc"] >= leak["label_window_leak"]["clean"]["pr_auc"]
    )
    mine.sweep(params, paths)
    mined = mine.run(params, paths)
    assert mined["itemsets_identical"] is True
    rules = json.loads(paths.rules_file.read_text())
    assert rules["header"]["rule_count"] == len(rules["rules"]) > 0
    ev = rec_eval.run(params, paths)
    assert ev["hit_rate_hybrid"] >= ev["hit_rate_popularity"]


def test_registry_resolves_champion(tmp_repo, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    paths, params = tmp_repo
    if not paths.model_file.exists():
        pytest.skip("pipeline test did not run")
    from boreal_marche.registry import load_retention_model

    monkeypatch.setenv("BOREAL_MODEL_SOURCE", "registry")
    _, source = load_retention_model(params, paths)
    assert source.startswith("models:/") or source.endswith("model.joblib")
    monkeypatch.setenv("BOREAL_MODEL_SOURCE", "local")
    _, source = load_retention_model(params, paths)
    assert Path(source).name == "model.joblib"
