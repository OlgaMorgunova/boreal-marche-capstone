"""Leakage investigation for Head A (D3 "Data integrity and leakage").

Three experiments, each reported as metric *with* the leak vs *without* it, on the validation
snapshot (never the test snapshot):

1. **Label-window leak** — features computed through the label window (``feature_end = T + 90d``).
2. **Split leak** — plain row-level ``KFold`` over stacked snapshots (the same customer on both
   sides of a fold) vs customer-grouped folds, compared with true temporal validation.
3. **Cancellation-netting leak** — how many pre-cutoff purchase lines a "net cancellations against
   the original invoice" cleaning step would delete because of cancellations that happen *after*
   the cutoff.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.features import build_all
from boreal_marche.retention.cv import (
    cross_validate,
    load_gold,
    make_folds,
    split_xy,
    summarise,
)
from boreal_marche.retention.models import build, score_all
from boreal_marche.utils import utc_now, write_json

MODEL = "xgboost"  # untuned default: the experiment isolates the data effect, not tuning


def label_window_leak(params: Params, paths: Paths, clean: pd.DataFrame) -> dict[str, Any]:
    lines = pd.read_parquet(paths.silver_lines)
    canc = pd.read_parquet(paths.silver_cancellations)
    leaky = build_all(params, lines, canc, feature_end_offset_days=params.windows.label_days)
    out = {}
    for label, gold in (("clean", clean), ("leaky", leaky)):
        tr, va = gold[gold["split"] == "train"], gold[gold["split"] == "validation"]
        Xtr, ytr = split_xy(tr)
        Xva, yva = split_xy(va)
        est = build(MODEL, params.seed).fit(Xtr, ytr)
        out[label] = score_all(yva, est.predict_proba(Xva)[:, 1])
    return out


def split_leak(params: Params, clean: pd.DataFrame) -> dict[str, Any]:
    tr = clean[clean["split"] == "train"].reset_index(drop=True)
    X, y = split_xy(tr)
    grouped = make_folds(tr, params.spotcheck.n_folds, params.seed)
    kf = KFold(n_splits=params.spotcheck.n_folds, shuffle=True, random_state=params.seed)
    naive = list(kf.split(X))
    shared = [
        int(np.isin(tr.loc[va, "customer_id"], tr.loc[trn, "customer_id"]).mean() * 1000) / 1000
        for trn, va in naive
    ]
    est = build(MODEL, params.seed)
    va = clean[clean["split"] == "validation"]
    Xva, yva = split_xy(va)
    temporal = score_all(yva, build(MODEL, params.seed).fit(X, y).predict_proba(Xva)[:, 1])
    return {
        "row_kfold": summarise(cross_validate(est, X, y, naive)),
        "grouped_kfold": summarise(cross_validate(est, X, y, grouped)),
        "temporal_validation": temporal,
        "share_of_val_rows_whose_customer_is_in_train_fold_rowkfold": shared,
    }


def netting_leak(params: Params, paths: Paths) -> list[dict[str, Any]]:
    lines = pd.read_parquet(
        paths.silver_lines, columns=["customer_id", "stock_code", "invoice_date", "has_customer"]
    )
    lines = lines[lines["has_customer"]]
    canc = pd.read_parquet(
        paths.silver_cancellations, columns=["customer_id", "stock_code", "invoice_date"]
    )
    canc = canc.dropna(subset=["customer_id"])
    rows = []
    cutoffs = (
        params.windows.cutoffs.train
        + params.windows.cutoffs.validation
        + params.windows.cutoffs.test
    )
    for c in cutoffs:
        t = pd.Timestamp(c)
        future_c = canc[
            (canc["invoice_date"] >= t)
            & (canc["invoice_date"] < t + pd.Timedelta(days=params.windows.label_days))
        ]
        past = lines[
            (lines["invoice_date"] < t)
            & (lines["invoice_date"] >= t - pd.Timedelta(days=params.windows.observation_days))
        ]
        keys = future_c[["customer_id", "stock_code"]].drop_duplicates()
        hit = past.merge(keys, on=["customer_id", "stock_code"], how="inner")
        rows.append(
            {
                "cutoff": c,
                "pre_cutoff_lines_that_netting_would_touch": int(len(hit)),
                "customers_affected": int(hit["customer_id"].nunique()),
            }
        )
    return rows


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    clean = load_gold(paths)
    payload: dict[str, Any] = {
        "generated_at": utc_now(),
        "model": MODEL,
        "label_window_leak": label_window_leak(params, paths, clean),
        "split_leak": split_leak(params, clean),
        "netting_leak": netting_leak(params, paths),
    }
    write_json(paths.metrics / "leakage.json", payload)
    lw = payload["label_window_leak"]
    print(
        f"label-window leak: clean PR-AUC {lw['clean']['pr_auc']:.4f} -> leaky "
        f"{lw['leaky']['pr_auc']:.4f}"
    )
    sl = payload["split_leak"]
    print(
        f"split: row KFold {sl['row_kfold']['pr_auc_mean']:.4f}, grouped "
        f"{sl['grouped_kfold']['pr_auc_mean']:.4f}, temporal "
        f"{sl['temporal_validation']['pr_auc']:.4f}"
    )
    print("netting:", payload["netting_leak"])
    return payload


if __name__ == "__main__":
    run()
