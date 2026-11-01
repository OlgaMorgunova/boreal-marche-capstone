"""Data-minimisation audit (D5; Law 25 necessity principle, GOVERNANCE_LOI25.md §5).

For each removable feature group, refit the champion's pipeline and hyper-parameters **without**
that group and measure the loss: PR-AUC over the same 5 customer-grouped folds of the training
snapshots (the D2 spot-check folds), and on the validation snapshot. The decision rule was written
before the ablation (GOVERNANCE §5): a group whose removal costs less than **0.005 PR-AUC** in CV
is not necessary and is removed from the model; ``region`` is kept for delivery, monitoring and the
fairness audit even when the model stops using it.

Writes ``reports/metrics/minimisation.json``.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from boreal_marche.config import Paths, load_params
from boreal_marche.features import FEATURES
from boreal_marche.retention.cv import load_gold, make_folds
from boreal_marche.retention.models import make_preprocessor
from boreal_marche.utils import utc_now, write_json

NECESSITY_MIN_LOSS = 0.005
GROUPS: dict[str, list[str]] = {
    "time_of_day": ["weekend_share_180d", "mean_order_hour_180d"],
    "region": ["region"],
    "cancellations": ["cancel_lines_180d", "cancel_rate_180d"],
    "category_mix": ["share_seasonal", "share_kitchen", "share_home_decor", "share_bags_wrap",
                     "share_lighting", "category_entropy_180d"],
    "basket_shape": ["lines_per_order_180d", "units_per_line_180d", "distinct_skus_180d",
                     "sku_repeat_rate_180d"],
    "outlier_lines": ["outlier_line_share_180d"],
    "lifetime_value": ["revenue_lifetime", "orders_lifetime"],
}  # fmt: skip
WHY = {
    "time_of_day": "When a person shops (weekend, hour) is behavioural profiling beyond what a "
    "re-order prediction needs.",
    "region": "Needed for delivery; as a model input it is a proxy for geography.",
    "cancellations": "Dissatisfaction signal.",
    "category_mix": "What kind of products a person buys.",
    "basket_shape": "Size and breadth of orders.",
    "outlier_lines": "Bulk-buying flag.",
    "lifetime_value": "Long-run value; grows with observable history (monitoring finding, D4).",
}


def pipeline_without(champion_pipe: Any, drop: list[str]) -> Any:
    """The champion's estimator and hyper-parameters on all 37 features minus ``drop``."""
    pipe = clone(champion_pipe)
    pipe.steps[0] = ("prep", make_preprocessor([f for f in FEATURES if f not in drop]))
    return pipe


def run() -> dict[str, Any]:
    params, paths = load_params(), Paths()
    champ = joblib.load(paths.model_file)
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    val = gold[gold["split"] == "validation"].reset_index(drop=True)
    folds = make_folds(train, params.spotcheck.n_folds, params.seed)
    y = train["y"].to_numpy()

    def score(drop: list[str]) -> dict[str, Any]:
        feats = [f for f in FEATURES if f not in drop]
        cv = []
        for tr, va in folds:
            pipe = pipeline_without(champ.pipeline, drop).fit(train.loc[tr, feats], y[tr])
            cv.append(
                average_precision_score(y[va], pipe.predict_proba(train.loc[va, feats])[:, 1])
            )
        pipe = pipeline_without(champ.pipeline, drop).fit(train[feats], y)
        v = average_precision_score(val["y"], pipe.predict_proba(val[feats])[:, 1])
        return {"cv_pr_auc": float(np.mean(cv)), "cv_folds": [float(x) for x in cv],
                "val_pr_auc": float(v), "n_features": len(feats)}  # fmt: skip

    full = score([])
    rows = []
    for name, cols in GROUPS.items():
        s = score(cols)
        loss = full["cv_pr_auc"] - s["cv_pr_auc"]
        fold_loss = np.array(full["cv_folds"]) - np.array(s["cv_folds"])
        rows.append({
            "group": name, "features": cols, "why_audited": WHY[name], **s,
            "cv_loss": float(loss), "val_loss": float(full["val_pr_auc"] - s["val_pr_auc"]),
            "folds_worse": int((fold_loss > 0).sum()),
            "necessary": bool(loss >= NECESSITY_MIN_LOSS),
        })  # fmt: skip
        print(f"{name:15s} CV loss {loss:+.4f}  val loss {rows[-1]['val_loss']:+.4f}")
    removable = [f for r in rows if not r["necessary"] for f in r["features"]]
    combined = score(removable) if removable else None
    out = {
        "generated_at": utc_now(),
        "rule": f"remove a group if its CV PR-AUC loss < {NECESSITY_MIN_LOSS}",
        "full": full,
        "groups": rows,
        "removable": removable,
        "combined_without_removable": combined,
    }
    write_json(paths.metrics / "minimisation.json", out)
    return out


if __name__ == "__main__":
    r = run()
    print("removable:", r["removable"])
    if r["combined_without_removable"]:
        c, full = r["combined_without_removable"], r["full"]
        print(f"all removable dropped: CV {c['cv_pr_auc']:.4f} (full {full['cv_pr_auc']:.4f}), "
              f"val {c['val_pr_auc']:.4f} (full {full['val_pr_auc']:.4f})")  # fmt: skip
