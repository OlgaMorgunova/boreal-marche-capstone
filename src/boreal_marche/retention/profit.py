"""Retention-campaign economics: turn a calibrated P(return) into a contact decision.

Action: send the 15% retention coupon to a customer. Per contacted customer (stated assumptions in
``params.yaml -> business``):

* true lapser (y = 0): with probability ``save_rate`` the coupon brings them back, earning
  ``value_i - discount_i``; the coupon always costs ``coupon_cost``;
* would-return-anyway (y = 1): pure loss ``coupon_cost + discount_i`` (margin given away).

``value_i = gross_margin * revenue_180d * value_horizon_days / 180`` (margin over the value horizon
if the customer resumes their recent buying rate) and
``discount_i = discount_rate * aov_180d`` — both computed from point-in-time features.
Not contacting has zero incremental profit, so "coupon to nobody" is the zero line and
"coupon to everyone" is the current quarterly policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from boreal_marche.config import BusinessParams


def customer_economics(
    df: pd.DataFrame, b: BusinessParams, horizon_days: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if horizon_days is not None:
        b = b.model_copy(update={"value_horizon_days": horizon_days})
    value = b.gross_margin * df["revenue_180d"].to_numpy(dtype=float) * b.value_horizon_days / 180.0
    discount = b.discount_rate * df["aov_180d"].to_numpy(dtype=float)
    return value, discount


def realised_profit(
    y: np.ndarray,
    value: np.ndarray,
    discount: np.ndarray,
    b: BusinessParams,
    save_rate: float | None = None,
) -> np.ndarray:
    """Profit of contacting each customer, given the true outcome (expected over the save)."""
    s = b.save_rate if save_rate is None else save_rate
    return np.where(y == 0, s * (value - discount) - b.coupon_cost, -(b.coupon_cost + discount))


def expected_profit(
    p_return: np.ndarray,
    value: np.ndarray,
    discount: np.ndarray,
    b: BusinessParams,
    save_rate: float | None = None,
) -> np.ndarray:
    """Expected profit of contacting each customer using the calibrated probability."""
    s = b.save_rate if save_rate is None else save_rate
    return (1 - p_return) * (s * (value - discount) - b.coupon_cost) - p_return * (
        b.coupon_cost + discount
    )


def profit_curve(
    p_return: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    discount: np.ndarray,
    b: BusinessParams,
    grid: np.ndarray | None = None,
    save_rate: float | None = None,
) -> pd.DataFrame:
    """Realised and expected campaign profit when contacting everyone with P(return) <= tau."""
    grid = np.round(np.linspace(0, 1, 101), 2) if grid is None else grid
    per_real = realised_profit(y, value, discount, b, save_rate)
    per_exp = expected_profit(p_return, value, discount, b, save_rate)
    rows = []
    for tau in grid:
        m = p_return <= tau
        rows.append(
            {
                "threshold": float(tau),
                "n_contacted": int(m.sum()),
                "share_contacted": float(m.mean()),
                "realised_profit": float(per_real[m].sum()),
                "expected_profit": float(per_exp[m].sum()),
                "lapsers_contacted": int(((y == 0) & m).sum()),
            }
        )
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame, column: str = "realised_profit") -> float:
    """Threshold with maximum profit; ties broken towards contacting fewer customers."""
    top = curve[column].max()
    return float(curve.loc[curve[column] == top, "threshold"].min())


def policy_summary(
    p_return: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    discount: np.ndarray,
    b: BusinessParams,
    threshold: float,
) -> dict[str, float]:
    per_real = realised_profit(y, value, discount, b)
    per_exp = expected_profit(p_return, value, discount, b)
    model = p_return <= threshold
    ev = per_exp > 0
    n = len(y)
    return {
        "n_customers": n,
        "threshold": threshold,
        "model_contacted": int(model.sum()),
        "model_profit": float(per_real[model].sum()),
        "ev_policy_contacted": int(ev.sum()),
        "ev_policy_profit": float(per_real[ev].sum()),
        "everyone_profit": float(per_real.sum()),
        "nobody_profit": 0.0,
        "model_vs_everyone": float(per_real[model].sum() - per_real.sum()),
        "model_profit_per_1000": float(per_real[model].sum() / n * 1000),
        "everyone_profit_per_1000": float(per_real.sum() / n * 1000),
        "lapsers_total": int((y == 0).sum()),
        "lapsers_contacted_model": int(((y == 0) & model).sum()),
    }
