"""Live monitoring report: input drift, prediction drift, data-quality drift and operations.

Everything is computed from the log tables against ``artifacts/reference_profile.json``:

* **Input drift** per feature — PSI on the training decile bins (+ a missing bin) and the two-sample
  Kolmogorov–Smirnov statistic against a fixed training sample (numeric features).
* **Prediction drift** — PSI of the live ``p_return`` against the champion's validation predictions,
  plus the mean probability and the share of customers proposed for a coupon.
* **Data-quality drift** — change in each feature's missing rate; share of ``/predict`` requests
  rejected by validation (422); share of unknown SKUs in baskets.
* **Operations** — request volume, latency percentiles per endpoint, error rate, and the
  recommendation rule fire-rate (share of non-empty baskets answered by a rule).

Each check gets a traffic light from ``params.yaml -> monitoring.thresholds``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, recall_score
from sqlalchemy import func, select

from boreal_marche.config import Band, MonitoringParams
from boreal_marche.monitoring.store import LogStore, prediction_log

STATUS_ORDER = {"insufficient_data": -1, "green": 0, "amber": 1, "red": 2}
EPS = 1e-4
QUIET_PATHS = ("/metrics", "/health", "/ready", "/monitoring", "/docs", "/openapi.json")


def band(value: float | None, b: Band, lower_is_worse: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "insufficient_data"
    if lower_is_worse:
        return "red" if value < b.red else "amber" if value < b.amber else "green"
    return "red" if value >= b.red else "amber" if value >= b.amber else "green"


def worst(statuses: list[str]) -> str:
    known = [s for s in statuses if s != "insufficient_data"]
    return max(known, key=STATUS_ORDER.__getitem__) if known else "insufficient_data"


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def numeric_shares(values: pd.Series, edges: list[float]) -> np.ndarray:
    v = pd.to_numeric(values, errors="coerce")
    present = v.dropna().to_numpy(dtype=float)
    counts = np.bincount(np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1)
    return np.append(counts, v.isna().sum()) / max(len(v), 1)


def feature_drift(name: str, ref: dict[str, Any], values: pd.Series, m: MonitoringParams) -> dict:
    t = m.thresholds
    missing_live = float(values.isna().mean())
    out: dict[str, Any] = {
        "feature": name,
        "missing_ref": ref["missing_rate"],
        "missing_live": missing_live,
        "missing_delta": abs(missing_live - ref["missing_rate"]),
    }
    if ref["type"] == "numeric":
        expected = np.append(ref["proportions"], ref["missing_rate"])
        out["psi"] = psi(expected, numeric_shares(values, ref["bin_edges"]))
        live = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if len(live) >= 2:
            ks = ks_2samp(np.asarray(ref["sample"]), live)
            out["ks"], out["ks_pvalue"] = float(ks.statistic), float(ks.pvalue)
        else:
            out["ks"], out["ks_pvalue"] = None, None
        out["mean_ref"], out["mean_live"] = ref["mean"], float(np.mean(live)) if len(live) else None
    else:
        cats = list(ref["categories"])
        live_share = values.astype("string").fillna("<NA>").value_counts(normalize=True)
        other = float(live_share[[c for c in live_share.index if c not in cats]].sum())
        expected = np.array([ref["categories"][c] for c in cats] + [0.0])
        actual = np.array([float(live_share.get(c, 0.0)) for c in cats] + [other])
        out["psi"], out["ks"], out["ks_pvalue"] = psi(expected, actual), None, None
    out["status"] = worst(
        [band(out["psi"], t.psi)] + ([band(out["ks"], t.ks)] if out.get("ks") is not None else [])
    )
    return out


def _percentiles(lat: pd.Series) -> dict[str, float | None]:
    if lat.empty:
        return {"p50": None, "p95": None, "p99": None}
    q = lat.quantile([0.5, 0.95, 0.99])
    return {"p50": float(q[0.5]), "p95": float(q[0.95]), "p99": float(q[0.99])}


def operations(req: pd.DataFrame, m: MonitoringParams) -> dict[str, Any]:
    if req.empty:
        return {"request_count": 0, "by_endpoint": {}, "error_rate": None, "rpm": None}
    traffic = req[~req["path"].str.startswith(QUIET_PATHS)]
    span_min = max((req["ts"].max() - req["ts"].min()) / 60, 1 / 60)
    by_ep = {}
    for path, g in traffic.groupby("path"):
        by_ep[str(path)] = {
            "count": len(g),
            **_percentiles(g["latency_ms"]),
            "error_rate": float((g["status"] >= 500).mean()),
            "client_error_rate": float(g["status"].between(400, 499).mean()),
        }
    return {
        "request_count": len(traffic),
        "rpm": len(traffic) / span_min,
        "error_rate": float((traffic["status"] >= 500).mean()) if len(traffic) else None,
        "by_endpoint": by_ep,
        "from": float(req["ts"].min()),
        "to": float(req["ts"].max()),
    }


def build_report(
    store: LogStore, profile: dict[str, Any], m: MonitoringParams, now: float | None = None
) -> dict[str, Any]:
    t = m.thresholds
    preds = store.recent("prediction_log", m.window_predictions)
    reqs = store.recent("request_log", m.window_requests)
    recs = store.recent("recommendation_log", m.window_predictions)
    chats = store.recent("chat_log", m.window_predictions)
    enough = len(preds) >= m.min_predictions
    checks: list[dict[str, Any]] = []

    # --- input drift --------------------------------------------------------------------------
    features: list[dict[str, Any]] = []
    if enough:
        X = pd.DataFrame([json.loads(f) for f in preds["features"]])
        for name, ref in profile["features"].items():
            if name in m.exclude_features or name not in X:
                continue
            features.append(feature_drift(name, ref, X[name], m))
    psis = [f["psi"] for f in features]
    share_red = float(np.mean([p >= t.psi.red for p in psis])) if psis else None
    input_status = (
        "insufficient_data"
        if share_red is None
        else "red"
        if share_red >= t.drifted_feature_share_red
        else "amber"
        if max(psis) >= t.psi.amber
        else "green"
    )
    checks.append(
        {
            "name": "input_drift",
            "status": input_status,
            "value": share_red,
            "unit": "share of features with PSI >= red",
            "threshold": {"amber": f"any PSI >= {t.psi.amber}", "red": t.drifted_feature_share_red},
            "detail": f"top drifted: {', '.join(f['feature'] for f in sorted(features, key=lambda f: -f['psi'])[:3])}"  # noqa: E501
            if features
            else None,
        }
    )

    # --- prediction drift ---------------------------------------------------------------------
    pred_ref = profile["prediction"]
    prediction: dict[str, Any] = {"mean_ref": pred_ref["mean"]}
    prediction["send_coupon_rate_ref"] = pred_ref["send_coupon_rate"]
    if enough:
        p = preds["p_return"].to_numpy(dtype=float)
        counts = np.bincount(
            np.searchsorted(pred_ref["bin_edges"], p, side="right"),
            minlength=len(pred_ref["bin_edges"]) + 1,
        )
        prediction |= {
            "psi": psi(np.asarray(pred_ref["proportions"]), counts / len(p)),
            "mean_live": float(p.mean()),
            "send_coupon_rate_live": float(preds["send_coupon"].astype(bool).mean()),
        }
    prediction["status"] = band(prediction.get("psi"), t.prediction_psi)
    checks.append(
        {
            "name": "prediction_drift",
            "status": prediction["status"],
            "value": prediction.get("psi"),
            "unit": "PSI of p_return",
            "threshold": t.prediction_psi.model_dump(),
        }
    )

    # --- data quality -------------------------------------------------------------------------
    predict_reqs = reqs[reqs["path"] == "/predict"] if len(reqs) else reqs
    quality: dict[str, Any] = {
        "max_missing_delta": max((f["missing_delta"] for f in features), default=None),
        "worst_missing_feature": max(features, key=lambda f: f["missing_delta"])["feature"]
        if features
        else None,
        "predict_validation_error_rate": float((predict_reqs["status"] == 422).mean())
        if len(predict_reqs)
        else None,
        "unknown_item_rate": float(recs["n_unknown"].sum() / max(recs["basket_size"].sum(), 1))
        if len(recs)
        else None,
    }
    quality_value = (
        None
        if quality["max_missing_delta"] is None
        else max(quality["max_missing_delta"], quality["predict_validation_error_rate"] or 0.0)
    )
    checks.append(
        {
            "name": "data_quality",
            "status": band(quality_value, t.missing_rate_delta),
            "value": quality_value,
            "unit": "max(missing-rate change, /predict 422 rate)",
            "threshold": t.missing_rate_delta.model_dump(),
        }
    )

    # --- operations ---------------------------------------------------------------------------
    ops = operations(reqs, m)
    enough_reqs = ops["request_count"] >= m.min_predictions
    checks.append(
        {
            "name": "error_rate",
            "status": band(ops["error_rate"] if enough_reqs else None, t.error_rate),
            "value": ops["error_rate"],
            "unit": "share of 5xx responses",
            "threshold": t.error_rate.model_dump(),
        }
    )
    p95 = ops["by_endpoint"].get("/predict", {}).get("p95")
    checks.append(
        {
            "name": "latency_p95_predict",
            "status": band(p95 if enough_reqs else None, t.p95_latency_ms),
            "value": p95,
            "unit": "ms, server-side",
            "threshold": t.p95_latency_ms.model_dump(),
        }
    )

    # --- recommendation rule fire-rate --------------------------------------------------------
    nonempty = recs[recs["basket_size"] > 0] if len(recs) else recs
    fire = (
        float(nonempty["strategy"].str.startswith("rules").mean()) if len(nonempty) >= 20 else None
    )
    recommend = {
        "n": len(recs),
        "rule_fire_rate": fire,
        "rule_fire_rate_ref": profile["recommend"]["rule_fire_rate"],
        "strategies": recs["strategy"].value_counts().to_dict() if len(recs) else {},
    }
    checks.append(
        {
            "name": "rule_fire_rate",
            "status": band(fire, t.rule_fire_rate, lower_is_worse=True),
            "value": fire,
            "unit": "share of non-empty baskets answered by a rule",
            "threshold": t.rule_fire_rate.model_dump() | {"direction": "lower is worse"},
        }
    )

    assistant: dict[str, Any] = {"n": len(chats)}
    if len(chats):
        assistant |= {
            "by_provider": chats.groupby("provider")["latency_ms"]
            .agg(["count", "median"])
            .to_dict("index"),
            "refusal_rate": float(chats["refused"].astype(bool).mean()),
            "handoff_rate": float(chats["handoff"].astype(bool).mean()),
            "tools": chats["tool"].value_counts().to_dict(),
        }

    cohorts = preds["cohort"].fillna("none").value_counts().to_dict() if len(preds) else {}
    return {
        "generated_at": now or time.time(),
        "status": worst([c["status"] for c in checks]),
        "window": {
            "predictions": len(preds),
            "requests": len(reqs),
            "min_predictions": m.min_predictions,
            "from": float(preds["ts"].min()) if len(preds) else None,
            "to": float(preds["ts"].max()) if len(preds) else None,
        },
        "checks": checks,
        "features": sorted(features, key=lambda f: -f["psi"]),
        "excluded_features": m.exclude_features,
        "prediction": prediction,
        "data_quality": quality,
        "operations": ops,
        "recommend": recommend,
        "assistant": assistant,
        "cohorts_in_window": cohorts,
        "reference": {
            "source": profile["source"],
            "model_version": profile["model_version"],
            "generated_at": profile["generated_at"],
        },
    }


def timeseries(store: LogStore, n: int = 5000, bucket_seconds: int = 60) -> list[dict[str, Any]]:
    """Per-minute request volume, p95 latency, error rate and mean prediction (dashboard charts)."""
    reqs = store.recent("request_log", n)
    if reqs.empty:
        return []
    reqs = reqs[~reqs["path"].str.startswith(QUIET_PATHS)]
    preds = store.recent("prediction_log", n)
    reqs = reqs.assign(bucket=(reqs["ts"] // bucket_seconds * bucket_seconds).astype(int))
    g = reqs.groupby("bucket")
    out = pd.DataFrame(
        {
            "requests": g.size(),
            "p95_ms": g["latency_ms"].quantile(0.95),
            "error_rate": g["status"].apply(lambda s: float((s >= 500).mean())),
        }
    )
    if len(preds):
        preds = preds.assign(bucket=(preds["ts"] // bucket_seconds * bucket_seconds).astype(int))
        out = out.join(preds.groupby("bucket")["p_return"].mean().rename("mean_p_return"))
        shifted = preds.assign(s=preds["cohort"].eq("shifted")).groupby("bucket")["s"].mean()
        out = out.join(shifted.rename("shifted_share"))
    out = out.reset_index().rename(columns={"bucket": "ts"})
    return json.loads(out.to_json(orient="records"))  # type: ignore[no-any-return]


def realised_metrics(store: LogStore, label_days: int) -> dict[str, Any]:
    df = store.labelled_predictions()
    out: dict[str, Any] = {
        "n_labelled": len(df),
        "label_window_days": label_days,
        "first_prediction_ts": None,
        "first_true_signal_ts": None,
    }
    with store.engine.connect() as conn:
        ts0 = conn.execute(select(func.min(prediction_log.c.ts))).scalar()
    if ts0 is not None:
        out["first_prediction_ts"] = float(ts0)
        out["first_true_signal_ts"] = float(ts0) + label_days * 86400 + 7 * 86400
    if len(df) >= 20 and df["label"].nunique() == 2:
        y = df["label"].to_numpy(dtype=int)
        p = df["p_return"].to_numpy(dtype=float)
        lapse = 1 - y
        out |= {
            "pr_auc": float(average_precision_score(y, p)),
            "lapser_recall": float(recall_score(lapse, df["send_coupon"].astype(int))),
            "base_rate": float(y.mean()),
            "mean_p_return": float(np.mean(p)),
        }
    return out
