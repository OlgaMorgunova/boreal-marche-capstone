"""Spot-check: every model family and both baselines under identical folds and preprocessing.

Folds: ``StratifiedGroupKFold`` over the stacked *train* snapshots, grouped by customer so the same
customer never appears on both sides of a fold (a customer has up to four snapshots). Validation and
test snapshots are not touched here.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from boreal_marche import tracking
from boreal_marche.config import Params, Paths, load_params
from boreal_marche.features import MODEL_FEATURES
from boreal_marche.reporting import style
from boreal_marche.retention.cv import cross_validate, load_gold, make_folds, split_xy, summarise
from boreal_marche.retention.models import baselines, build, is_baseline, model_zoo
from boreal_marche.utils import utc_now, write_json


def boxplot(results: dict[str, list[dict[str, float]]], path: str) -> None:
    order = sorted(results, key=lambda k: np.median([f["pr_auc"] for f in results[k]]))
    data = [[f["pr_auc"] for f in results[k]] for k in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.6)
    ax.set_yticks(range(1, len(order) + 1), order)
    for patch, name in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(style.NEUTRAL if is_baseline(name) else style.BLUE)
    ax.set_xlabel("PR-AUC per fold (5-fold StratifiedGroupKFold, train snapshots)")
    ax.set_title("Head A spot-check: fold PR-AUC by model (grey = baselines)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, object]:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    X, y = split_xy(train)
    folds = make_folds(train, params.spotcheck.n_folds, params.seed)
    tracking.setup(params, paths, params.mlflow.experiment_retention)
    names = list(baselines()) + list(model_zoo(params.seed))
    results: dict[str, list[dict[str, float]]] = {}
    table: list[dict[str, object]] = []
    with mlflow.start_run(
        run_name="spotcheck", tags=tracking.lineage_tags(paths) | {"stage": "spotcheck"}
    ):
        mlflow.log_params(
            {
                "n_folds": len(folds),
                "n_rows": len(train),
                "n_features": len(MODEL_FEATURES),
                "cutoffs": ",".join(params.windows.cutoffs.train),
            }
        )
        for name in names:
            est = build(name, params.seed)
            fm = cross_validate(est, X, y, folds)
            results[name] = fm
            s = summarise(fm)
            table.append({"model": name, **s})
            print(
                f"{name:<26} PR-AUC {s['pr_auc_mean']:.4f} ± {s['pr_auc_std']:.4f}  "
                f"ROC {s['roc_auc_mean']:.4f}  Brier {s['brier_mean']:.4f}  "
                f"fit {s['fit_seconds_mean']:.2f}s"
            )
            with mlflow.start_run(run_name=f"spot_{name}", nested=True):
                mlflow.set_tag("model_family", name)
                mlflow.log_metrics(s)
        fig_path = paths.figures / "spotcheck_boxplot.png"
        boxplot(results, str(fig_path))
        mlflow.log_artifact(str(fig_path))
    ranked = sorted(
        (r for r in table if not is_baseline(str(r["model"]))),
        key=lambda r: -float(r["pr_auc_mean"]),  # type: ignore[arg-type]
    )
    top2 = [str(r["model"]) for r in ranked[:2]]
    payload: dict[str, object] = {
        "generated_at": utc_now(),
        "n_rows": len(train),
        "n_folds": len(folds),
        "positive_rate": float(y.mean()),
        "table": table,
        "folds": results,
        "top2": top2,
    }
    write_json(paths.metrics / "spotcheck.json", payload)
    print("Top-2 carried forward:", top2)
    return payload


if __name__ == "__main__":
    run()
