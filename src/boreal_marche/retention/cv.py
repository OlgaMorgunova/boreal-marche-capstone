"""Cross-validation helpers shared by the spot-check, tuning and the SageMaker entry point.

Deliberately free of MLflow/matplotlib imports so the SageMaker training container stays small.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from boreal_marche.config import Paths
from boreal_marche.features import FEATURES, TARGET
from boreal_marche.retention.models import score_all

Folds = list[tuple[np.ndarray, np.ndarray]]


def load_gold(paths: Paths) -> pd.DataFrame:
    return pd.read_parquet(paths.gold / "retention.parquet")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df[FEATURES].reset_index(drop=True), df[TARGET].to_numpy()


def make_folds(train: pd.DataFrame, n_folds: int, seed: int) -> Folds:
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(train[FEATURES], train[TARGET], groups=train["customer_id"]))


def cross_validate(
    estimator: object,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    on_fold: Callable[[int, float], None] | None = None,
) -> list[dict[str, float]]:
    """Fit/score on each fold; ``on_fold(i, running_mean_pr_auc)`` enables Optuna pruning."""
    out: list[dict[str, float]] = []
    for i, (tr, va) in enumerate(folds):
        est = clone(estimator)
        t0 = time.perf_counter()
        est.fit(X.iloc[tr], y[tr])  # type: ignore[attr-defined]
        fit_s = time.perf_counter() - t0
        p = est.predict_proba(X.iloc[va])[:, 1]  # type: ignore[attr-defined]
        m = score_all(y[va], p) | {"fit_seconds": fit_s}
        out.append(m)
        if on_fold is not None:
            on_fold(i, float(np.mean([f["pr_auc"] for f in out])))
    return out


def summarise(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    df = pd.DataFrame(fold_metrics)
    res: dict[str, float] = {}
    for c in df.columns:
        res[f"{c}_mean"] = float(df[c].mean())
        res[f"{c}_std"] = float(df[c].std(ddof=1))
    return res
