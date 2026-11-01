"""Head A subgroup fairness audit and mitigations (D5; GOVERNANCE_LOI25.md §7).

**Audit.** For each subgroup of region, tenure band and spend quartile: selection rate (share
proposed for the coupon), lapser recall (share of true lapsers proposed — the course's "recall"),
coupon precision (share of coupons that reach a lapser), PR-AUC and calibration-in-the-large. A
group is **flagged** when its lapser recall is more than 5 points below the overall recall.
Spend-quartile edges are fixed on the training snapshots (``revenue_180d``), so a per-group policy
is servable.

**Mitigations** — chosen on the validation snapshot only, then each scored once on the test snapshot
after the choice was fixed (reported, never used to choose):

* ``M1_expected_value`` — contact when the expected campaign profit is positive (``profit.py``):
  value-aware, so high spenders are contacted at a higher P(return);
* ``M2_group_thresholds`` — one threshold per spend quartile, set on validation so every quartile's
  lapser recall reaches the overall recall of the current policy (equal opportunity);
* ``M3_reweighted`` — retrain with sample weights that up-weight lapsers of under-served groups
  (inverse of each group's validation recall under the current policy), same features and
  hyper-parameters, recalibrated, same threshold.

Removing ``region`` from the model was evaluated by the minimisation audit (``minimise.py``) and is
done; the audit of the 37-feature v0.9 model is kept in ``fairness_v09_37features.json``.

For every policy: overall recall, worst gap, flagged groups, coupons sent, PR-AUC and realised
campaign profit under the stated assumptions. Writes ``reports/metrics/fairness.json``.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from boreal_marche.config import BusinessParams, Paths, load_params
from boreal_marche.retention.artifact import Calibrator, RetentionModel
from boreal_marche.retention.cv import load_gold
from boreal_marche.retention.profit import customer_economics, expected_profit, realised_profit
from boreal_marche.utils import read_json, utc_now, write_json

GAP_LIMIT = 0.05
MIN_LAPSERS = 30
TENURE_BINS = [-1, 180, 365, np.inf]
TENURE_LABELS = ["<6m", "6-12m", ">12m"]
SPEND_LABELS = ["Q1 (low)", "Q2", "Q3", "Q4 (top)"]


def spend_edges(train: pd.DataFrame) -> list[float]:
    """Inner quartile edges of ``revenue_180d`` on the training snapshots."""
    return [float(x) for x in np.quantile(train["revenue_180d"], [0.25, 0.5, 0.75])]


def subgroups(df: pd.DataFrame, edges: list[float]) -> dict[str, pd.Series]:
    return {
        "region": df["region"].astype(str),
        "tenure_band": pd.cut(df["tenure_days"], TENURE_BINS, labels=TENURE_LABELS).astype(str),
        "spend_quartile": pd.cut(
            df["revenue_180d"], [-np.inf, *edges, np.inf], labels=SPEND_LABELS
        ).astype(str),
    }


def audit(df: pd.DataFrame, p: np.ndarray, send: np.ndarray, edges: list[float]) -> dict[str, Any]:
    y = df["y"].to_numpy()
    lapser = y == 0

    def stats(m: np.ndarray) -> dict[str, Any]:
        n_l = int((m & lapser).sum())
        sent = int((m & send).sum())
        return {
            "n": int(m.sum()),
            "n_lapsers": n_l,
            "lapse_rate": float(lapser[m].mean()),
            "selection_rate": float(send[m].mean()),
            "lapser_recall": float(send[m & lapser].mean()) if n_l else None,
            "coupon_precision": float(lapser[m & send].mean()) if sent else None,
            "pr_auc": float(average_precision_score(y[m], p[m]))
            if 0 < y[m].sum() < m.sum() and m.sum() >= 50 else None,
            "mean_p": float(p[m].mean()),
            "return_rate": float(y[m].mean()),
        }  # fmt: skip

    overall = stats(np.ones(len(df), bool))
    groups = []
    for dim, labels in subgroups(df, edges).items():
        for g in sorted(labels.unique()):
            s = stats((labels == g).to_numpy())
            rec = s["lapser_recall"]
            gap = None if rec is None else overall["lapser_recall"] - rec
            s |= {"dimension": dim, "group": g, "gap": gap,
                  "reliable": s["n_lapsers"] >= MIN_LAPSERS,
                  "flagged": bool(gap is not None and gap > GAP_LIMIT
                                  and s["n_lapsers"] >= MIN_LAPSERS)}  # fmt: skip
            groups.append(s)
    reliable = [g["gap"] for g in groups if g["reliable"] and g["gap"] is not None]
    return {
        "overall": overall,
        "worst_gap": max(reliable, default=0.0),
        "n_flagged": sum(g["flagged"] for g in groups),
        "flagged": [f"{g['dimension']}={g['group']}" for g in groups if g["flagged"]],
        "groups": groups,
    }


def summarise(
    name: str, df: pd.DataFrame, p: np.ndarray, send: np.ndarray, edges: list[float],
    b: BusinessParams,
) -> dict[str, Any]:  # fmt: skip
    value, discount = customer_economics(df, b)
    y = df["y"].to_numpy()
    profit = realised_profit(y, value, discount, b)
    a = audit(df, p, send, edges)
    sq = {g["group"]: g for g in a["groups"] if g["dimension"] == "spend_quartile"}
    top = sq.get("Q4 (top)", {})
    missed_top = (df["revenue_180d"].to_numpy() > edges[-1]) & (y == 0) & ~send
    return {
        "policy": name,
        "pr_auc": float(average_precision_score(y, p)),
        "coupons": int(send.sum()),
        "profit": float(profit[send].sum()),
        "lapser_recall": a["overall"]["lapser_recall"],
        "coupon_precision": a["overall"]["coupon_precision"],
        "worst_gap": a["worst_gap"],
        "flagged": a["flagged"],
        "recall_by_spend": {k: v["lapser_recall"] for k, v in sq.items()},
        "top_quartile_recall": top.get("lapser_recall"),
        "missed_top_lapsers": int(missed_top.sum()),
        "missed_top_value": float(value[missed_top].sum()),
        "audit": a,
    }


def group_thresholds(
    df: pd.DataFrame, p: np.ndarray, base_threshold: float, edges: list[float]
) -> dict[str, float]:
    """Per spend quartile: the lowest threshold whose lapser recall reaches the overall recall of
    the single-threshold policy (never lower than that threshold: nobody loses the offer)."""
    lapser = df["y"].to_numpy() == 0
    target = float((p[lapser] <= base_threshold).mean())
    q = subgroups(df, edges)["spend_quartile"].to_numpy()
    out = {}
    for g in SPEND_LABELS:
        pl = np.sort(p[(q == g) & lapser])
        k = int(np.ceil(target * len(pl))) - 1
        out[g] = float(max(base_threshold, pl[max(k, 0)])) if len(pl) else base_threshold
    return out


def apply_group_thresholds(df: pd.DataFrame, p: np.ndarray, th: dict[str, float],
                           edges: list[float]) -> np.ndarray:  # fmt: skip
    q = subgroups(df, edges)["spend_quartile"].to_numpy()
    return p <= np.array([th[g] for g in q])


def retrain(
    champion: RetentionModel, train: pd.DataFrame, val: pd.DataFrame, features: list[str],
    weights: np.ndarray | None,
) -> RetentionModel:  # fmt: skip
    """Same pipeline and hyper-parameters as the champion, refit on the training snapshots."""
    from sklearn.base import clone

    from boreal_marche.retention.models import make_preprocessor

    pipe = clone(champion.pipeline)
    pipe.steps[0] = ("prep", make_preprocessor(features))
    fit_kw = {} if weights is None else {"model__sample_weight": weights}
    pipe.fit(train[features], train["y"].to_numpy(), **fit_kw)
    cal = Calibrator(champion.calibrator.method).fit(
        pipe.predict_proba(val[features])[:, 1], val["y"].to_numpy()
    )
    return RetentionModel(pipe, cal, features, champion.threshold, champion.version + "-d5")


def reweight(train: pd.DataFrame, val_recall: dict[tuple[str, str], float], overall: float,
             edges: list[float]) -> np.ndarray:  # fmt: skip
    """Lapser weight = product over dimensions of (overall recall / group recall), capped at 3."""
    w = np.ones(len(train))
    lapser = train["y"].to_numpy() == 0
    for dim, labels in subgroups(train, edges).items():
        for g, rec in ((g, r) for (d, g), r in val_recall.items() if d == dim):
            m = (labels == g).to_numpy() & lapser
            w[m] *= min(3.0, overall / max(rec, 1e-3)) if rec < overall else 1.0
    return w


def run() -> dict[str, Any]:
    params, paths = load_params(), Paths()
    b = params.business
    gold = load_gold(paths)
    split = {
        s: gold[gold["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")
    }
    train, val, test = split["train"], split["validation"], split["test"]
    edges = spend_edges(train)
    champ: RetentionModel = joblib.load(paths.model_file)
    tau = champ.threshold

    def p_of(m: RetentionModel, df: pd.DataFrame) -> np.ndarray:
        return m.predict_proba(df[m.feature_names])[:, 1]

    pv, pt = p_of(champ, val), p_of(champ, test)
    base_val = audit(val, pv, pv <= tau, edges)

    # M1: expected-value policy.
    def ev_send(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
        value, discount = customer_economics(df, b)
        return expected_profit(p, value, discount, b) > 0

    # M2: per-quartile thresholds fitted on validation.
    th = group_thresholds(val, pv, tau, edges)
    # M3: lapser re-weighting from the validation recall gaps of the current policy.
    val_recall = {(g["dimension"], g["group"]): g["lapser_recall"]
                  for g in base_val["groups"] if g["reliable"]}  # fmt: skip
    w = reweight(train, val_recall, base_val["overall"]["lapser_recall"], edges)
    m3 = retrain(champ, train, val, champ.feature_names, w)

    policies: dict[str, dict[str, Any]] = {}
    for snap, df, p in (("validation", val, pv), ("test", test, pt)):
        p3 = p_of(m3, df)
        policies[snap] = {
            "M0_single_threshold": summarise("M0_single_threshold", df, p, p <= tau, edges, b),
            "M1_expected_value": summarise("M1_expected_value", df, p, ev_send(df, p), edges, b),
            "M2_group_thresholds": summarise(
                "M2_group_thresholds", df, p, apply_group_thresholds(df, p, th, edges), edges, b
            ),
            "M3_reweighted": summarise("M3_reweighted", df, p3, p3 <= tau, edges, b),
        }
    prev = read_json(paths.metrics / "test_evaluation.json").get("n_test_looks")
    out = {
        "generated_at": utc_now(),
        "model_version": champ.version,
        "threshold": tau,
        "spend_edges_train": edges,
        "gap_limit": GAP_LIMIT,
        "min_lapsers": MIN_LAPSERS,
        "group_thresholds_M2": th,
        "reweight_summary": {
            "min": float(w.min()),
            "max": float(w.max()),
            "share_upweighted": float((w > 1).mean()),
        },  # fmt: skip
        "test_use": f"scored once after the choice on validation; earlier test looks: {prev}",
        "policies": policies,
    }
    write_json(paths.metrics / "fairness.json", out)
    return out


if __name__ == "__main__":
    r = run()
    for snap, pol in r["policies"].items():
        print(f"\n{snap}")
        for name, s in pol.items():
            rb = " ".join(
                f"{k[:2]}={v:.2f}" for k, v in s["recall_by_spend"].items() if v is not None
            )
            print(f"  {name:22s} PR-AUC {s['pr_auc']:.3f} coupons {s['coupons']:5d} "
                  f"profit {s['profit']:9.0f} recall {s['lapser_recall']:.3f} "
                  f"worst gap {s['worst_gap']:.3f} [{rb}] flagged {s['flagged']}")  # fmt: skip
