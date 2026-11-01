"""Training entry point that runs identically on a laptop and as an Amazon SageMaker job.

SageMaker script-mode contract:

* hyper-parameters arrive as command-line arguments (``--n-estimators 400 ...``);
* the training channel is mounted at ``SM_CHANNEL_TRAIN`` (defaults to ``data/gold`` locally);
* the model must be written to ``SM_MODEL_DIR`` (defaults to ``artifacts/sm_model`` locally);
* the objective metric is printed as ``cv_pr_auc=<value>`` so the HPO job can parse it with the
  regex ``cv_pr_auc=([0-9\\.]+)`` (SageMaker), and logged with MLflow when ``--log-mlflow`` is
  given (Azure ML sweep jobs read metrics from MLflow).

The same file is the Azure ML command-job entry point (``--train``/``--model-dir`` are passed as
arguments there instead of environment variables).

Local run:  ``python -m boreal_marche.retention.sm_train --n-estimators 300 --max-depth 4``
The script depends only on the gold Parquet (uploaded to S3 by the launcher), not on MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from boreal_marche.features import MODEL_FEATURES, TARGET
from boreal_marche.retention.cv import cross_validate, make_folds
from boreal_marche.retention.models import build


def int_arg(value: str) -> int:
    """Accept "300" and "300.0" (Azure ML's quniform sampler emits floats)."""
    return int(float(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-estimators", type=int_arg, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int_arg, default=4)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.9)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/gold"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "artifacts/sm_model"))
    ap.add_argument(
        "--log-mlflow",
        action="store_true",
        help="also log the objective with MLflow (required by Azure ML sweep jobs)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> float:
    a = parse_args(argv)
    gold = pd.read_parquet(Path(a.train) / "retention.parquet")
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    X, y = train[MODEL_FEATURES], train[TARGET].to_numpy()
    est = build("xgboost", a.seed)
    est.set_params(
        model__n_estimators=a.n_estimators,
        model__learning_rate=a.learning_rate,
        model__max_depth=a.max_depth,
        model__min_child_weight=a.min_child_weight,
        model__subsample=a.subsample,
        model__colsample_bytree=a.colsample_bytree,
        model__reg_lambda=a.reg_lambda,
    )
    t0 = time.perf_counter()
    folds = make_folds(train, a.n_folds, a.seed)
    score = float(np.mean([m["pr_auc"] for m in cross_validate(est, X, y, folds)]))
    est.fit(X, y)
    wall = time.perf_counter() - t0
    out = Path(a.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(est, out / "pipeline.joblib")
    (out / "metrics.json").write_text(
        json.dumps({"cv_pr_auc": score, "wall_seconds": wall, "args": vars(a)}, indent=2)
    )
    if a.log_mlflow:
        import mlflow

        mlflow.log_metric("cv_pr_auc", score)
        mlflow.log_metric("wall_seconds", wall)
    print(f"cv_pr_auc={score:.6f}")
    print(f"wall_seconds={wall:.2f}")
    return score


if __name__ == "__main__":
    main()
