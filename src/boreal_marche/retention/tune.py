"""Bayesian hyper-parameter search (Optuna TPE + median pruning) for the spot-check top-2.

Pruning works across folds: after each fold the running mean PR-AUC is reported, and a trial whose
running mean is below the median of earlier trials at the same fold is stopped. Every trial
(complete or pruned) is logged to MLflow as a nested run.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna

from boreal_marche import tracking
from boreal_marche.config import Params, Paths, load_params
from boreal_marche.reporting import style
from boreal_marche.retention.cv import (
    cross_validate,
    load_gold,
    make_folds,
    split_xy,
    summarise,
)
from boreal_marche.retention.models import build
from boreal_marche.utils import read_json, utc_now, write_json


def suggest(name: str, t: optuna.trial.BaseTrial) -> dict[str, Any]:
    """Search spaces; prefixed with ``model__`` for the pipeline."""
    if name == "extra_trees" or name == "random_forest":
        depth = t.suggest_int("max_depth", 4, 32)
        p = {
            "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": depth,
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 60, log=True),
            "max_features": t.suggest_float("max_features", 0.2, 1.0),
            "criterion": t.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    elif name == "xgboost":
        p = {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "max_depth": t.suggest_int("max_depth", 2, 10),
            "min_child_weight": t.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": t.suggest_float("gamma", 0.0, 5.0),
        }
    elif name == "lightgbm":
        p = {
            "n_estimators": t.suggest_int("n_estimators", 100, 1000, step=50),
            "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "num_leaves": t.suggest_int("num_leaves", 7, 127, log=True),
            "min_child_samples": t.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": t.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": t.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    elif name == "hist_gb":
        p = {
            "learning_rate": t.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 7, 127, log=True),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 200, log=True),
            "l2_regularization": t.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            "max_iter": t.suggest_int("max_iter", 100, 800, step=50),
        }
    elif name == "logreg":
        p = {"C": t.suggest_float("C", 1e-3, 100.0, log=True)}
    elif name == "mlp":
        width = t.suggest_int("width", 16, 256, log=True)
        p = {
            "hidden_layer_sizes": (width, max(8, width // 2)),
            "alpha": t.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "learning_rate_init": t.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    elif name == "gaussian_nb":
        p = {"var_smoothing": t.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)}
    elif name == "knn":
        p = {
            "n_neighbors": t.suggest_int("n_neighbors", 5, 300, log=True),
            "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
        }
    elif name == "decision_tree":
        p = {
            "max_depth": t.suggest_int("max_depth", 2, 20),
            "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 200, log=True),
            "criterion": t.suggest_categorical("criterion", ["gini", "entropy"]),
        }
    else:
        raise ValueError(f"no search space for {name!r}")
    return {f"model__{k}": v for k, v in p.items()}


def _plot_history(study: optuna.Study, default: float, name: str, path: str) -> None:
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    xs = [t.number for t in done]
    ys = [float(t.value) for t in done if t.value is not None]
    best = np.maximum.accumulate(ys) if ys else np.array([])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(xs, ys, s=14, label="completed trial (CV mean PR-AUC)")
    ax.plot(xs, best, color=style.ORANGE, label="best so far")
    pruned = [t.number for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    if pruned:
        ax.scatter(
            pruned,
            [min(ys + [default]) - 0.002] * len(pruned),
            marker="x",
            color=style.NEUTRAL,
            s=14,
            label="pruned trial",
        )
    ax.axhline(default, ls="--", color=style.INK_2, label=f"untuned default ({default:.4f})")
    ax.set_xlabel("trial")
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Optuna optimisation history — {name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(imp: dict[str, float], name: str, path: str) -> None:
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh([k for k, _ in items], [v for _, v in items], color=style.BLUE)
    ax.set_xlabel("relative importance (fANOVA)")
    ax.set_title(f"Hyper-parameter importance — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tune_one(name: str, params: Params, paths: Paths, default_score: float) -> dict[str, Any]:
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    X, y = split_xy(train)
    folds = make_folds(train, params.spotcheck.n_folds, params.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"tune_{name}",
        sampler=optuna.samplers.TPESampler(seed=params.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
    )

    def objective(trial: optuna.Trial) -> float:
        overrides = suggest(name, trial)
        est = build(name, params.seed)
        est.set_params(**overrides)

        def on_fold(i: int, running: float) -> None:
            trial.report(running, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        with mlflow.start_run(run_name=f"{name}_trial_{trial.number:03d}", nested=True):
            mlflow.log_params({k.removeprefix("model__"): v for k, v in overrides.items()})
            try:
                fm = cross_validate(est, X, y, folds, on_fold=on_fold)
            except optuna.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise
            s = summarise(fm)
            mlflow.set_tag("trial_state", "COMPLETE")
            mlflow.log_metrics(s)
            trial.set_user_attr("summary", s)
            return s["pr_auc_mean"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(
        run_name=f"tune_{name}",
        tags=tracking.lineage_tags(paths) | {"stage": "tuning", "model_family": name},
    ):
        study.optimize(
            objective, n_trials=params.tuning.n_trials, timeout=params.tuning.timeout_seconds
        )
        best = study.best_trial
        try:
            importances = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=params.seed)
            )
        except (RuntimeError, ValueError):
            importances = {}
        hist_png = paths.figures / f"optuna_history_{name}.png"
        imp_png = paths.figures / f"optuna_importance_{name}.png"
        _plot_history(study, default_score, name, str(hist_png))
        if importances:
            _plot_importance(importances, name, str(imp_png))
            mlflow.log_artifact(str(imp_png))
        mlflow.log_artifact(str(hist_png))
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metrics(
            {
                "best_pr_auc": float(best.value or 0.0),
                "default_pr_auc": default_score,
                "improvement": float(best.value or 0.0) - default_score,
            }
        )
    states = [t.state.name for t in study.trials]
    return {
        "model": name,
        "n_trials": len(study.trials),
        "n_complete": states.count("COMPLETE"),
        "n_pruned": states.count("PRUNED"),
        "best_value": float(best.value or 0.0),
        "best_params": best.params,
        "best_summary": best.user_attrs.get("summary", {}),
        "default_value": default_score,
        "improvement": float(best.value or 0.0) - default_score,
        "importances": {k: float(v) for k, v in importances.items()},
        "history": [
            {"number": t.number, "state": t.state.name, "value": t.value, "params": t.params}
            for t in study.trials
        ],
    }


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    spot = read_json(paths.metrics / "spotcheck.json")
    defaults = {r["model"]: r["pr_auc_mean"] for r in spot["table"]}
    tracking.setup(params, paths, params.mlflow.experiment_retention)
    results = {}
    for name in spot["top2"]:
        r = tune_one(name, params, paths, defaults[name])
        results[name] = r
        print(
            f"{name}: default {r['default_value']:.4f} -> best {r['best_value']:.4f} "
            f"(+{r['improvement']:.4f}); {r['n_complete']} complete, {r['n_pruned']} pruned"
        )
    winner = max(results.values(), key=lambda r: r["best_value"])
    payload = {"generated_at": utc_now(), "results": results, "winner": winner["model"]}
    write_json(paths.metrics / "tuning.json", payload)
    print("Winner:", winner["model"])
    return payload


if __name__ == "__main__":
    run()
