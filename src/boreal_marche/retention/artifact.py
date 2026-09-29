"""The single shippable Head A artifact: preprocessing + estimator + calibrator + threshold.

Preprocessing lives *inside* the artifact so the API never re-implements it (no training/serving
skew). The object is serialised with joblib; see D3 "Artifact discipline" for the pickle risks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class Calibrator:
    """Maps raw scores to probabilities: ``none`` | ``sigmoid`` (Platt on logit) | ``isotonic``."""

    def __init__(self, method: str = "sigmoid") -> None:
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(method)
        self.method = method
        self._model: Any = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6).fit(_logit(p_raw).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(
                p_raw, y
            )
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if self.method == "sigmoid":
            return np.asarray(self._model.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1])
        if self.method == "isotonic":
            return np.asarray(self._model.predict(p_raw))
        return np.asarray(p_raw)


@dataclass
class RetentionModel:
    """Calibrated P(customer places another order within the label window)."""

    pipeline: Any
    calibrator: Calibrator
    feature_names: list[str]
    threshold: float
    version: str

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        return X[self.feature_names]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(self._frame(X))[:, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.calibrator.transform(self.predict_raw(X))
        return np.column_stack([1 - p, p])

    def decide(self, X: pd.DataFrame) -> np.ndarray:
        """True = send the retention coupon (calibrated P(return) at or below the threshold)."""
        return self.predict_proba(X)[:, 1] <= self.threshold
