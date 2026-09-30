"""Shared preprocessing, model zoo, baselines and metrics for Head A.

Every candidate is wrapped in the *same* preprocessing pipeline (``make_preprocessor``) so the
spot-check compares estimators, not preprocessing choices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from boreal_marche.features import CATEGORICAL, NUMERIC

N_JOBS = 8


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Monotone compression of heavy tails; tree models are invariant to it, linear models gain."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def model_zoo(seed: int) -> dict[str, Callable[[], BaseEstimator]]:
    """Ten model families with sensible, untuned defaults."""
    return {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "gaussian_nb": lambda: GaussianNB(),
        "knn": lambda: KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=N_JOBS),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=50, random_state=seed
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=N_JOBS, random_state=seed
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=N_JOBS, random_state=seed
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(random_state=seed),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=N_JOBS,
            random_state=seed,
            verbose=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=N_JOBS,
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=seed,
        ),
    }


class RecencyRule(ClassifierMixin, BaseEstimator):
    """One-rule business heuristic: "ordered in the last ``days`` days -> will come back".

    Probabilities are the empirical return rates of the two groups in the training fold, so the
    rule can be scored with the same metrics (including Brier) as the models.
    """

    def __init__(self, days: float = 60.0) -> None:
        self.days = days

    def _recent(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(X["recency_days"].to_numpy() <= self.days)

    def fit(self, X: pd.DataFrame, y: Any) -> RecencyRule:
        y = np.asarray(y)
        r = self._recent(X)
        self.classes_ = np.array([0, 1])
        self.rate_recent_ = float(y[r].mean()) if r.any() else float(y.mean())
        self.rate_other_ = float(y[~r].mean()) if (~r).any() else float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.where(self._recent(X), self.rate_recent_, self.rate_other_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._recent(X).astype(int)


class RecencyDeciles(ClassifierMixin, BaseEstimator):
    """Stronger single-feature heuristic: return rate per recency decile of the training fold.

    Not a model a manager would write on a napkin, but it answers the sharper question
    "how much does machine learning add beyond sorting customers by recency?".
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y: Any) -> RecencyDeciles:
        y = np.asarray(y)
        r = X["recency_days"].to_numpy()
        self.classes_ = np.array([0, 1])
        self.edges_ = np.unique(np.quantile(r, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        b = np.searchsorted(self.edges_, r, side="right")
        self.rates_ = np.array(
            [y[b == i].mean() if (b == i).any() else y.mean() for i in range(len(self.edges_) + 1)]
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        b = np.searchsorted(self.edges_, X["recency_days"].to_numpy(), side="right")
        p = self.rates_[b]
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def baselines() -> dict[str, Callable[[], BaseEstimator]]:
    return {
        "baseline_dummy_prior": lambda: DummyClassifier(strategy="prior"),
        "baseline_rule_recency60": lambda: RecencyRule(60.0),
        "baseline_recency_deciles": lambda: RecencyDeciles(10),
    }


def is_baseline(name: str) -> bool:
    return name.startswith("baseline_")


def build(name: str, seed: int, **overrides: Any) -> BaseEstimator:
    """Return a full pipeline (baselines are returned bare: they read raw columns)."""
    if is_baseline(name):
        return baselines()[name]()
    est = model_zoo(seed)[name]()
    if overrides:
        est.set_params(**overrides)
    return make_pipeline(est)


def recall_at_top(y: np.ndarray, score: np.ndarray, frac: float = 0.1) -> float:
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    pos = y.sum()
    return float(y[top].sum() / pos) if pos else float("nan")


def precision_at_top(y: np.ndarray, score: np.ndarray, frac: float = 0.1) -> float:
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-score, kind="stable")[:k]
    return float(y[top].mean())


def score_all(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Metrics for P(return). ``pr_auc_lapse`` scores the action class (lapsers) for reference."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "recall_top_decile": recall_at_top(y, p),
        "precision_top_decile": precision_at_top(y, p),
        "pr_auc_lapse": float(average_precision_score(1 - y, 1 - p)),
    }
