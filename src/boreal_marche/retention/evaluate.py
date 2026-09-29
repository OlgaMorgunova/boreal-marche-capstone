"""One-time held-out test evaluation and error analysis for Head A.

The test snapshot (cutoff 2011-09-01) is scored exactly once. A guard file records the evaluation;
re-running requires ``--force`` and the number of looks is incremented and reported, so the report
can state honestly how many times the test set was viewed.
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.reporting import style
from boreal_marche.retention.artifact import RetentionModel
from boreal_marche.retention.cv import load_gold, split_xy
from boreal_marche.retention.models import baselines, score_all
from boreal_marche.retention.profit import customer_economics, policy_summary
from boreal_marche.retention.train import ece
from boreal_marche.utils import read_json, utc_now, write_json


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for k, v in score_all(y[idx], p[idx]).items():
            draws.setdefault(k, []).append(v)
    return {
        k: {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
        for k, v in draws.items()
    }


def segment_table(
    df: pd.DataFrame, p: np.ndarray, threshold: float, train: pd.DataFrame
) -> list[dict[str, Any]]:
    """Per-segment quality. Segment edges (spend quartiles) come from the training snapshots."""
    d = df.copy()
    d["p"] = p
    d["contact"] = p <= threshold
    q = train["revenue_180d"].quantile([0.25, 0.5, 0.75]).to_numpy()
    d["spend_quartile"] = pd.cut(
        d["revenue_180d"], [-np.inf, *q, np.inf], labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    ).astype(str)
    d["tenure_band"] = pd.cut(
        d["tenure_days"],
        [-np.inf, 180, 365, np.inf],
        labels=["<6 months", "6-12 months", ">12 months"],
    ).astype(str)
    d["frequency_band"] = pd.cut(
        d["orders_180d"], [0, 1, 3, np.inf], labels=["1 order", "2-3 orders", "4+ orders"]
    ).astype(str)
    rows: list[dict[str, Any]] = []
    overall_lapse = d["y"] == 0
    overall_recall = float((d["contact"] & overall_lapse).sum() / max(1, overall_lapse.sum()))
    for seg in ["region", "tenure_band", "spend_quartile", "frequency_band"]:
        for val, g in d.groupby(seg):
            y = g["y"].to_numpy()
            lapse = y == 0
            n_lapse = int(lapse.sum())
            recall = float((g["contact"].to_numpy() & lapse).sum() / n_lapse) if n_lapse else None
            prec = (
                float((g["contact"].to_numpy() & lapse).sum() / g["contact"].sum())
                if g["contact"].any()
                else None
            )
            m = score_all(y, g["p"].to_numpy()) if 0 < y.mean() < 1 and len(g) >= 20 else {}
            rows.append(
                {
                    "segment": seg,
                    "value": str(val),
                    "n": int(len(g)),
                    "return_rate": float(y.mean()),
                    "mean_p": float(g["p"].mean()),
                    "pr_auc": m.get("pr_auc"),
                    "brier": m.get("brier"),
                    "lapse_recall": recall,
                    "lapse_precision": prec,
                    "selection_rate": float(g["contact"].mean()),
                    "recall_gap_vs_overall": (recall - overall_recall)
                    if recall is not None
                    else None,
                }
            )
    return rows


def run(
    params: Params | None = None,
    paths: Paths | None = None,
    force: bool = False,
    skip_if_done: bool = False,
) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    guard = paths.metrics / "test_evaluation.json"
    looks = 0
    if guard.exists():
        looks = int(read_json(guard).get("n_test_looks", 1))
        if skip_if_done and not force:
            print(
                f"Test set already evaluated ({looks} look(s)); keeping the recorded result in "
                f"{guard.name}. To re-score it on this machine run the evaluate step with --force "
                "(this adds a look, which must be reported)."
            )
            return dict(read_json(guard))
        if not force:
            raise SystemExit(
                f"Test set already evaluated ({looks} look(s)); use --force to "
                "re-evaluate — the number of looks is recorded in the report."
            )
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"].reset_index(drop=True)
    test = gold[gold["split"] == "test"].reset_index(drop=True)
    Xtr, ytr = split_xy(train)
    Xte, yte = split_xy(test)
    model: RetentionModel = joblib.load(paths.model_file)
    p = model.predict_proba(Xte)[:, 1]
    p_raw = model.predict_raw(Xte)
    metrics = score_all(yte, p)
    metrics["ece"] = ece(yte, p)
    ci = bootstrap_ci(yte, p, seed=params.seed)
    base = {}
    for name, factory in baselines().items():
        est = factory().fit(Xtr, ytr)
        pb = est.predict_proba(Xte)[:, 1]
        base[name] = score_all(yte, pb) | {"ci": bootstrap_ci(yte, pb, n=300, seed=params.seed)}
    value, discount = customer_economics(test, params.business)
    policy = policy_summary(p, yte, value, discount, params.business, model.threshold)
    segments = segment_table(test, p, model.threshold, train)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=style.NEUTRAL, label="perfect calibration")
    for label, pp in (("raw", p_raw), (f"calibrated ({model.calibrator.method})", p)):
        frac, mean = calibration_curve(yte, pp, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=4, label=f"{label} (ECE {ece(yte, pp):.3f})")
    ax.set_xlabel("mean predicted P(return)")
    ax.set_ylabel("observed return rate")
    ax.set_title("Reliability on the held-out test snapshot (2011-09-01)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.figures / "reliability_test.png", dpi=150)
    plt.close(fig)

    payload = {
        "generated_at": utc_now(),
        "n_test_looks": looks + 1,
        "cutoff": params.windows.cutoffs.test,
        "n": int(len(yte)),
        "positive_rate": float(yte.mean()),
        "model_version": model.version,
        "metrics": metrics,
        "ci95": ci,
        "raw_metrics": score_all(yte, p_raw),
        "baselines": base,
        "policy": policy,
        "segments": segments,
    }
    write_json(guard, payload)
    meta = read_json(paths.model_meta)
    meta["test_metrics"] = {
        "cutoff": params.windows.cutoffs.test[0],
        **metrics,
        "pr_auc_ci95": ci["pr_auc"],
        "evaluated_at": payload["generated_at"],
    }
    write_json(paths.model_meta, meta)
    print(
        f"TEST PR-AUC {metrics['pr_auc']:.4f} [{ci['pr_auc']['lo']:.4f}, {ci['pr_auc']['hi']:.4f}]"
        f" ROC {metrics['roc_auc']:.4f} Brier {metrics['brier']:.4f} ECE {metrics['ece']:.3f}"
    )
    for k, v in base.items():
        print(f"  {k:<26} PR-AUC {v['pr_auc']:.4f}")
    print("policy:", {k: round(v, 1) if isinstance(v, float) else v for k, v in policy.items()})
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-evaluate (increments the look count)")
    ap.add_argument(
        "--skip-if-done",
        action="store_true",
        help="exit successfully without scoring if the test set was already evaluated "
        "(used by the full pipeline on a fresh clone)",
    )
    ns = ap.parse_args()
    run(force=ns.force, skip_if_done=ns.skip_if_done)
