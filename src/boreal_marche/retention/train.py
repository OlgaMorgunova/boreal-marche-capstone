"""Final Head A fit: tuned model -> calibration -> profit threshold -> artifact -> MLflow registry.

Data usage (test snapshot is NOT touched here):

* train snapshots: fit the tuned pipeline;
* validation snapshot: choose the calibration method (5-fold out-of-fold Brier), choose the profit
  threshold on out-of-fold calibrated probabilities, then fit the final calibrator on all of it.
"""

from __future__ import annotations

import platform
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import lightgbm
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from mlflow.tracking import MlflowClient
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

import boreal_marche
from boreal_marche import tracking
from boreal_marche.config import Params, Paths, load_params
from boreal_marche.features import (
    COLLECTED,
    EXCLUDED_FROM_MODEL,
    FEATURES,
    MODEL_CATEGORICAL,
    MODEL_FEATURES,
)
from boreal_marche.reporting import style
from boreal_marche.retention.artifact import Calibrator, RetentionModel
from boreal_marche.retention.cv import load_gold, split_xy
from boreal_marche.retention.models import baselines, build, score_all
from boreal_marche.retention.profit import (
    best_threshold,
    customer_economics,
    policy_summary,
    profit_curve,
)
from boreal_marche.retention.tune import suggest
from boreal_marche.utils import git_commit, read_json, utc_now, write_json

METHODS = ["none", "sigmoid", "isotonic"]


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def tuned_pipeline(name: str, best_params: dict[str, Any], seed: int) -> Any:
    """Rebuild the pipeline from the Optuna trial parameters via the same ``suggest`` mapping."""
    import optuna

    trial = optuna.trial.FixedTrial(best_params)
    est = build(name, seed)
    est.set_params(**suggest(name, trial))
    return est


def oof_calibration(p_raw: np.ndarray, y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    out = {m: np.zeros_like(p_raw) for m in METHODS}
    for tr, va in cv.split(p_raw.reshape(-1, 1), y):
        for m in METHODS:
            out[m][va] = Calibrator(m).fit(p_raw[tr], y[tr]).transform(p_raw[va])
    return out


def reliability_plot(y: np.ndarray, curves: dict[str, np.ndarray], title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=style.NEUTRAL, label="perfect calibration")
    for label, p in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(
            mean,
            frac,
            marker="o",
            ms=4,
            label=f"{label} (Brier {brier_score_loss(y, p):.4f}, ECE {ece(y, p):.3f})",
        )
    ax.set_xlabel("mean predicted P(return)")
    ax.set_ylabel("observed return rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def profit_plot(curves: dict[str, pd.DataFrame], chosen: float, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, c in curves.items():
        ax.plot(c["threshold"], c["realised_profit"], label=label)
    main = next(iter(curves.values()))
    ax.plot(
        main["threshold"],
        main["expected_profit"],
        ls=":",
        color=style.INK_2,
        label="expected profit from calibrated P (base case)",
    )
    ax.axhline(0, color=style.NEUTRAL, lw=0.8)
    ax.axvline(chosen, color=style.ORANGE, ls="--", label=f"chosen threshold {chosen:.2f}")
    ax.set_xlabel("contact customers with calibrated P(return) <= threshold")
    ax.set_ylabel("campaign profit on validation snapshot (CAD)")
    ax.set_title("Expected-profit curve (threshold = 1.0 is today's coupon-to-everyone)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class RetentionPyfunc(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
    """MLflow wrapper so the registry can serve the joblib artifact unchanged."""

    def load_context(self, context: Any) -> None:
        self.model: RetentionModel = joblib.load(context.artifacts["model"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> np.ndarray:
        return self.model.predict_proba(model_input)[:, 1]


def fit_candidate(
    name: str, best_params: dict[str, Any], params: Params, train: pd.DataFrame, val: pd.DataFrame
) -> dict[str, Any]:
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    pipe = tuned_pipeline(name, best_params, params.seed).fit(Xtr, ytr)
    p_raw = pipe.predict_proba(Xva)[:, 1]
    oof = oof_calibration(p_raw, yva, params.seed)
    calib = {
        m: {
            "brier": float(brier_score_loss(yva, p)),
            "log_loss": float(log_loss(yva, p)),
            "ece": ece(yva, p),
        }
        for m, p in oof.items()
    }
    method = min(METHODS, key=lambda m: (round(calib[m]["brier"], 4), METHODS.index(m)))
    return {
        "name": name,
        "pipeline": pipe,
        "p_raw": p_raw,
        "oof": oof,
        "calibration": calib,
        "method": method,
        "val_metrics_raw": score_all(yva, p_raw),
        "val_metrics_cal": score_all(yva, oof[method]),
    }


def promotion_decision(
    cand: dict[str, Any],
    baseline_val: dict[str, dict[str, float]],
    current: tuple[float, int] | None,
) -> dict[str, Any]:
    """Written promotion policy (docs/MLFLOW_PROMOTION_POLICY.md), evaluated mechanically.

    ``current`` = (validation PR-AUC, number of input features) of the registry champion. Gate 4
    has a data-minimisation clause (D5): a candidate on strictly fewer features may replace the
    champion if it loses at most 0.005 PR-AUC — necessity outranks a marginal metric gain."""
    best_base = max(v["pr_auc"] for v in baseline_val.values())
    m = cand["val_metrics_cal"]
    beats = current is None or m["pr_auc"] >= current[0] + 0.005
    minimises = current is not None and (
        len(MODEL_FEATURES) < current[1] and m["pr_auc"] >= current[0] - 0.005
    )
    checks = {
        "beats_best_baseline_by_0.05_pr_auc": m["pr_auc"] >= best_base + 0.05,
        "calibrated_brier_not_worse_than_raw": cand["calibration"][cand["method"]]["brier"]
        <= cand["calibration"]["none"]["brier"],
        "ece_at_most_0.05": cand["calibration"][cand["method"]]["ece"] <= 0.05,
        "beats_current_champion_by_0.005_or_minimises": beats or minimises,
    }
    return {"checks": checks, "promote": all(checks.values()), "best_baseline_pr_auc": best_base}


def write_golden(paths: Paths, val: pd.DataFrame, n: int = 8) -> None:
    """Golden-prediction regression fixture: fixed validation customers and the champion's output.

    Validation rows (never test rows) are used so that regenerating the fixture is not a test look.
    """
    model: RetentionModel = joblib.load(paths.model_file)
    sample = val.sort_values("customer_id").iloc[:: max(1, len(val) // n)].head(n)
    p = model.predict_proba(sample[FEATURES])[:, 1]
    cases = []
    for (_, row), pi in zip(sample.iterrows(), p, strict=True):
        feats = {
            k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
            for k, v in row[COLLECTED].items()  # what the API stores; the rest is optional
        }
        cases.append(
            {"ref": f"val-{int(row['customer_id'])}", "features": feats, "p_return": float(pi)}
        )
    write_json(
        paths.root / "tests" / "golden" / "golden_predictions.json",
        {"model_version": model.version, "generated_at": utc_now(), "cases": cases},
    )


def registry_champion(client: MlflowClient, name: str) -> tuple[float, int] | None:
    """(validation PR-AUC, number of input features) of the ``champion`` alias, if any. Runs
    logged before D5 have no ``n_features`` parameter: they used all 37 features."""
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    run = client.get_run(mv.run_id)
    pr = run.data.metrics.get("val_cal_pr_auc")
    return None if pr is None else (pr, int(run.data.params.get("n_features", len(FEATURES))))


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    b = params.business
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    val = gold[gold["split"] == "validation"].reset_index(drop=True)
    Xtr, ytr = split_xy(train)
    Xva, yva = split_xy(val)
    tuning = read_json(paths.metrics / "tuning.json")
    ranked = sorted(tuning["results"].values(), key=lambda r: -r["best_value"])

    baseline_val = {}
    for bname, factory in baselines().items():
        est = factory().fit(Xtr, ytr)
        baseline_val[bname] = score_all(yva, est.predict_proba(Xva)[:, 1])

    tracking.setup(params, paths, params.mlflow.experiment_retention)
    client = MlflowClient()
    reg_name = params.mlflow.registered_model
    value, discount = customer_economics(val, b)
    candidates = [fit_candidate(r["model"], r["best_params"], params, train, val) for r in ranked]
    champion = candidates[0]
    y_oof = champion["oof"][champion["method"]]

    curves = {
        f"save_rate={s:.2f}": profit_curve(y_oof, yva, value, discount, b, save_rate=s)
        for s in b.sensitivity_save_rates
    }
    base_curve = profit_curve(y_oof, yva, value, discount, b)
    threshold = best_threshold(base_curve)
    sens = {
        k: {"best_threshold": best_threshold(c), "max_profit": float(c["realised_profit"].max())}
        for k, c in curves.items()
    }
    val_policy = policy_summary(y_oof, yva, value, discount, b, threshold)
    # First design (value horizon = one 90-day label window), kept as a documented negative result.
    v90, d90 = customer_economics(val, b, horizon_days=90)
    curve90 = profit_curve(y_oof, yva, v90, d90, b)
    horizon90 = {
        "best_threshold": best_threshold(curve90),
        "max_profit": float(curve90["realised_profit"].max()),
        "everyone_profit": float(curve90["realised_profit"].iloc[-1]),
    }
    profit_plot(
        {
            "realised, save_rate=0.20 (base)": base_curve,
            **{f"realised, {k}": c for k, c in curves.items() if k != "save_rate=0.20"},
        },
        threshold,
        str(paths.figures / "profit_curve_validation.png"),
    )
    reliability_plot(
        yva,
        {
            "raw (uncalibrated)": champion["p_raw"],
            "sigmoid (out-of-fold)": champion["oof"]["sigmoid"],
            "isotonic (out-of-fold)": champion["oof"]["isotonic"],
        },
        f"Reliability on validation snapshot — {champion['name']}",
        str(paths.figures / "reliability_validation.png"),
    )

    results: dict[str, Any] = {
        "generated_at": utc_now(),
        "baselines_validation": baseline_val,
        "candidates": {},
        "threshold": threshold,
        "threshold_sensitivity": sens,
        "validation_policy": val_policy,
        "horizon_90_negative_result": horizon90,
        "profit_curve": base_curve.to_dict("records"),
    }
    current_champion = registry_champion(client, reg_name)
    for rank, cand in enumerate(candidates):
        final_cal = Calibrator(cand["method"]).fit(cand["p_raw"], yva)
        model = RetentionModel(
            pipeline=cand["pipeline"],
            calibrator=final_cal,
            feature_names=list(MODEL_FEATURES),
            threshold=threshold,
            version=f"{boreal_marche.__version__}+{cand['name']}",
        )
        is_champion = rank == 0
        out_path = (
            paths.model_file if is_champion else paths.artifacts / f"model_{cand['name']}.joblib"
        )
        joblib.dump(model, out_path, compress=3)
        decision = promotion_decision(cand, baseline_val, current_champion)
        with mlflow.start_run(
            run_name=f"final_{cand['name']}",
            tags=tracking.lineage_tags(paths) | {"stage": "final", "model_family": cand["name"]},
        ) as run:
            mlflow.log_params(
                {
                    "calibration": cand["method"],
                    "threshold": threshold,
                    "n_train_rows": len(train),
                    "n_val_rows": len(val),
                    "n_features": len(MODEL_FEATURES),
                    **{
                        f"hp_{k}": v
                        for k, v in tuning["results"][cand["name"]]["best_params"].items()
                    },
                }
            )
            mlflow.log_metrics(
                {f"val_raw_{k}": v for k, v in cand["val_metrics_raw"].items()}
                | {f"val_cal_{k}": v for k, v in cand["val_metrics_cal"].items()}
            )
            for f in ("reliability_validation.png", "profit_curve_validation.png"):
                mlflow.log_artifact(str(paths.figures / f))
            info = mlflow.pyfunc.log_model(
                "model",
                python_model=RetentionPyfunc(),
                artifacts={"model": str(out_path)},
                input_example=Xva.head(3).astype({c: "object" for c in MODEL_CATEGORICAL}),
            )
            run_id = run.info.run_id
        mv = mlflow.register_model(info.model_uri, reg_name)
        alias = "champion" if is_champion and decision["promote"] else "challenger"
        client.set_registered_model_alias(reg_name, alias, mv.version)
        if alias == "champion":
            current_champion = (cand["val_metrics_cal"]["pr_auc"], len(MODEL_FEATURES))
        meta = {
            "model_version": model.version,
            "registry": {"name": reg_name, "version": int(mv.version), "alias": alias},
            "trained_at": utc_now(),
            "n_train_rows": len(train),
            "n_calibration_rows": len(val),
            "train_cutoffs": params.windows.cutoffs.train,
            "calibration_cutoffs": params.windows.cutoffs.validation,
            "feature_names": list(MODEL_FEATURES),
            "excluded_features": list(EXCLUDED_FROM_MODEL),
            "estimator": cand["name"],
            "hyperparameters": tuning["results"][cand["name"]]["best_params"],
            "calibration_method": cand["method"],
            "decision_threshold": threshold,
            "decision_rule": "contact (send coupon) if calibrated P(return) <= threshold",
            "validation_metrics_calibrated": cand["val_metrics_cal"],
            "test_metrics": None,
            "library_versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "lightgbm": lightgbm.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "joblib": joblib.__version__,
                "mlflow": mlflow.__version__,
            },
            "git_commit": git_commit(),
            "mlflow_run_id": run_id,
            "raw_sha256": read_json(paths.manifest)["sha256"],
            "promotion": decision,
        }
        meta_path = (
            paths.model_meta if is_champion else paths.artifacts / f"model_meta_{cand['name']}.json"
        )
        write_json(meta_path, meta)
        results["candidates"][cand["name"]] = {
            "calibration": cand["calibration"],
            "method": cand["method"],
            "val_metrics_raw": cand["val_metrics_raw"],
            "val_metrics_cal": cand["val_metrics_cal"],
            "promotion": decision,
            "alias": alias,
            "registry_version": int(mv.version),
            "mlflow_run_id": run_id,
        }
        print(
            f"{cand['name']}: calib={cand['method']} val PR-AUC(cal)="
            f"{cand['val_metrics_cal']['pr_auc']:.4f} alias={alias}"
        )
    results["champion"] = champion["name"]
    write_json(paths.metrics / "train_final.json", results)
    write_golden(paths, val)
    print(f"threshold={threshold:.2f}  validation policy: {val_policy}")
    return results


if __name__ == "__main__":
    run()
