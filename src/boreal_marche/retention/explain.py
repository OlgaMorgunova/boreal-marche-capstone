"""Head A interpretability (D5): exact per-prediction factors and the global importance study.

**Per prediction.** XGBoost computes TreeSHAP contributions natively (``pred_contribs=True``): for
each customer, one additive contribution per *transformed* column, in log-odds, that sums with the
bias to the raw margin exactly. The Platt calibrator is affine in that space
(``logit p = a * margin + b``), so multiplying by ``a`` gives contributions that sum exactly to the
**calibrated** log-odds the API returns. Transformed columns are folded back to the 37 input
features (missing indicators into their feature, one-hot levels into ``region`` / ``season_q``), so
a factor is always something the customer could recognise. The API serves this as
``POST /predict?explain=N`` (Law 25 s. 12.1: "the reasons and principal factors").

**Global study** (``python -m boreal_marche.retention.explain``, validation snapshot only — the test
snapshot is not touched): mean |contribution| and permutation importance (drop in PR-AUC, 5 repeats)
per feature, additivity error, direction checks against the expected sign, and the stability of the
top-3 factors under a ±5% perturbation of the numeric inputs. Writes
``reports/metrics/interpretability.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from boreal_marche.config import Paths, load_params
from boreal_marche.features import CATEGORICAL, FEATURE_SPEC, NUMERIC
from boreal_marche.retention.artifact import RetentionModel
from boreal_marche.retention.cv import load_gold
from boreal_marche.utils import utc_now, write_json

LABELS = {f.name: f.definition for f in FEATURE_SPEC}
# Constant within a quarter: its contribution is the season's baseline, not a reason about the
# customer (zero permutation importance on one snapshot), so served explanations leave it out.
CONTEXT_FEATURES = ("season_q",)
# Direction a domain reader expects for a *higher* value of the feature (+ = more likely to return).
EXPECTED_SIGN = {"recency_days": -1, "orders_90d": 1, "orders_180d": 1, "active_months_180d": 1,
                 "overdue_ratio": -1, "revenue_180d": 1}  # fmt: skip


def _source_feature(column: str) -> str:
    if column.startswith("missingindicator_"):
        return column.removeprefix("missingindicator_")
    for c in CATEGORICAL:
        if column.startswith(f"{c}_"):
            return c
    return column


def _calibration_slope(model: RetentionModel) -> tuple[float, float]:
    if model.calibrator.method != "sigmoid":
        raise ValueError("exact factors need the sigmoid calibrator (affine in log-odds)")
    lr = model.calibrator._model
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def contributions(model: RetentionModel, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-feature contributions to the calibrated log-odds of returning, and the base value.

    ``contrib.sum(axis=1) + base == logit(model.predict_proba(X)[:, 1])`` (up to float error)."""
    prep, est = model.pipeline[:-1], model.pipeline[-1]
    Z = np.asarray(prep.transform(model._frame(X)), dtype=np.float32)
    raw = est.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
    a, b = _calibration_slope(model)
    cols = [_source_feature(c) for c in prep.get_feature_names_out()]
    per_col = pd.DataFrame(a * raw[:, :-1], columns=cols)
    contrib = per_col.T.groupby(level=0, sort=False).sum().T.reindex(columns=model.feature_names)
    return contrib, a * raw[:, -1] + b


def top_factors(
    model: RetentionModel, X: pd.DataFrame, k: int = 3, exclude: tuple[str, ...] = CONTEXT_FEATURES
) -> list[list[dict[str, Any]]]:
    """The k largest contributions per row, signed: positive raises P(return)."""
    contrib, _ = contributions(model, X)
    contrib[[f for f in exclude if f in contrib]] = 0.0
    names = model.feature_names
    values = X[names].to_numpy(dtype=object)
    out = []
    for row, vals in zip(contrib.to_numpy(), values, strict=True):
        factors = []
        for j in np.argsort(-np.abs(row))[:k]:
            v = vals[j]
            factors.append({
                "feature": names[j],
                "label": LABELS[names[j]],
                "value": None if pd.isna(v) else v if isinstance(v, str) else float(v),
                "effect": round(float(row[j]), 4),
                "direction": "raises P(return)" if row[j] > 0 else "lowers P(return)",
            })  # fmt: skip
        out.append(factors)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def permutation_importance(
    model: RetentionModel, X: pd.DataFrame, y: np.ndarray, repeats: int, seed: int
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    base = average_precision_score(y, model.predict_proba(X)[:, 1])
    out = {}
    for f in model.feature_names:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = X[f].to_numpy()[rng.permutation(len(X))]
            drops.append(base - average_precision_score(y, model.predict_proba(Xp)[:, 1]))
        out[f] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return out


def stability(model: RetentionModel, X: pd.DataFrame, seed: int, noise: float = 0.05) -> dict:
    """Share of customers whose top-3 factor *set* survives a ±noise relative perturbation."""
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for f in (f for f in NUMERIC if f in model.feature_names):
        v = X[f].to_numpy(dtype=float)
        Xp[f] = v * (1 + rng.uniform(-noise, noise, len(v)))
    ta, tb = top_factors(model, X, 3), top_factors(model, Xp, 3)
    pairs = [
        ({d["feature"] for d in x}, {d["feature"] for d in y}) for x, y in zip(ta, tb, strict=True)
    ]
    return {
        "noise": noise,
        "top3_set_identical": float(np.mean([a == b for a, b in pairs])),
        "top3_mean_jaccard": float(np.mean([len(a & b) / len(a | b) for a, b in pairs])),
        "top1_identical": float(
            np.mean([x[0]["feature"] == y[0]["feature"] for x, y in zip(ta, tb, strict=True)])
        ),  # fmt: skip
    }


def run() -> dict[str, Any]:
    p, paths = load_params(), Paths()
    import joblib

    model: RetentionModel = joblib.load(paths.model_file)
    gold = load_gold(paths)
    val = gold[gold["split"] == "validation"].reset_index(drop=True)
    X, y = val[model.feature_names], val["y"].to_numpy()
    contrib, base = contributions(model, X)
    p_hat = model.predict_proba(X)[:, 1]
    additivity = np.abs(contrib.sum(axis=1).to_numpy() + base - _logit(p_hat))
    mean_abs = contrib.abs().mean().sort_values(ascending=False)
    perm = permutation_importance(model, X, y, repeats=5, seed=p.seed)
    signs = {}
    for f, s in ((f, s) for f, s in EXPECTED_SIGN.items() if f in model.feature_names):
        c = np.corrcoef(X[f].fillna(X[f].median()), contrib[f])[0, 1]
        signs[f] = {
            "expected": s,
            "corr_value_vs_effect": float(c),
            "agrees": bool(np.sign(c) == s),
        }
    tops = top_factors(model, X, 3)
    with_context = top_factors(model, X, 3, exclude=())
    season_in_top3 = float(
        np.mean([any(d["feature"] == "season_q" for d in t) for t in with_context])
    )
    top1 = pd.Series([t[0]["feature"] for t in tops]).value_counts(normalize=True)
    share_in_top3 = pd.Series([d["feature"] for t in tops for d in t]).value_counts() / len(tops)
    send = p_hat <= model.threshold
    out = {
        "generated_at": utc_now(),
        "snapshot": "validation",
        "n": len(val),
        "model_version": model.version,
        "method": "XGBoost TreeSHAP (pred_contribs) x Platt slope, folded to input features",
        "calibration_slope": _calibration_slope(model)[0],
        "base_logit": float(base[0]),
        "base_p": float(1 / (1 + np.exp(-base[0]))),
        "additivity_max_abs_error": float(additivity.max()),
        "global": [
            {
                "feature": f,
                "label": LABELS[f],
                "mean_abs_effect": float(mean_abs[f]),
                "perm_pr_auc_drop": perm[f]["mean"],
                "perm_std": perm[f]["std"],
                "share_top1": float(top1.get(f, 0.0)),
                "share_top3": float(share_in_top3.get(f, 0.0)),
            }
            for f in mean_abs.index
        ],  # fmt: skip
        "direction_checks": signs,
        "stability": stability(model, X, p.seed),
        "n_features_ever_top3": int((share_in_top3 > 0).sum()),
        "season_q_in_top3_if_not_excluded": season_in_top3,
        "examples": [
            {
                "p_return": float(p_hat[i]),
                "send_coupon": bool(send[i]),
                "y": int(y[i]),
                "factors": tops[i],
            }
            for i in (
                int(np.argmin(p_hat)),
                int(np.argmin(np.abs(p_hat - model.threshold))),
                int(np.argmax(p_hat)),
            )
        ],  # fmt: skip
    }
    write_json(paths.metrics / "interpretability.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print(f"additivity error {r['additivity_max_abs_error']:.2e}; top features:")
    for g in r["global"][:8]:
        print(f"  {g['feature']:24s} |effect| {g['mean_abs_effect']:.3f}  "
              f"perm dPR-AUC {g['perm_pr_auc_drop']:.4f}  top1 {g['share_top1']:.0%}")  # fmt: skip
    print(r["stability"])
