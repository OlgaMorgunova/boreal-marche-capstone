"""Offline build of the serving-side reference artifacts (``make reference``).

* ``artifacts/reference_profile.json`` — the distribution the live drift checks compare against:
  decile bins + missing rate per feature (PSI), a fixed-size sample per feature (KS), the champion's
  prediction distribution (prediction drift) and the offline rule fire-rate (Head B coverage on the
  temporal holdout).

  **Which snapshot is the reference.** The reference is the most recent snapshot of the model's
  training pipeline — the calibration snapshot (``windows.cutoffs.validation``, on which the
  calibrator and the threshold were fitted) — minus the demo customers, who are held out to play
  the "normal" live traffic. The pooled training snapshots were tried first and rejected: real
  traffic from the next quarter was already red on several features, because lifetime features
  (``tenure_days``, ``orders_lifetime``, ``revenue_lifetime``) and cadence missingness grow by
  construction as the observable history lengthens (the data start in December 2009, so early
  snapshots are left-censored). A monitor that is red on healthy traffic gets ignored. That known,
  explained shift is recorded under ``known_shift_vs_pooled_training`` instead of being alerted on.
* ``artifacts/demo_customers.json`` — pseudonymised feature vectors of validation-snapshot
  customers. They stand in for the online feature store: the UI, the assistant's retention tool
  and the traffic simulator look customers up here. No labels are shipped.
* ``artifacts/catalogue.json`` — product catalogue export (SKU, description, category, median price,
  popularity) used by the product-lookup tool and the basket picker.
"""

from __future__ import annotations

import hashlib
from typing import Any

import joblib
import numpy as np
import pandas as pd

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.features import CATEGORICAL, FEATURES, NUMERIC
from boreal_marche.monitoring.drift import feature_drift
from boreal_marche.retention.cv import load_gold
from boreal_marche.utils import git_commit, read_json, utc_now, write_json

N_BINS = 10
N_DEMO_CUSTOMERS = 300


def customer_ref(customer_id: int) -> str:
    """Stable pseudonym; the raw identifier never leaves the offline pipeline."""
    return "BM-" + hashlib.sha256(f"boreal:{customer_id}".encode()).hexdigest()[:8].upper()


def numeric_profile(values: pd.Series, sample_rows: int, seed: int) -> dict[str, Any]:
    v = values.astype("float64")
    present = v.dropna().to_numpy()
    edges = np.unique(np.quantile(present, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1)
    rng = np.random.default_rng(seed)
    sample = rng.choice(present, size=min(sample_rows, len(present)), replace=False)
    return {
        "type": "numeric",
        # Interior edges; bin i holds edges[i-1] < x <= edges[i] (searchsorted side="right").
        "bin_edges": edges.tolist(),
        "proportions": (counts / len(v)).tolist(),
        "missing_rate": float(v.isna().mean()),
        "mean": float(present.mean()),
        "std": float(present.std()),
        "sample": np.round(np.sort(sample), 6).tolist(),
    }


def categorical_profile(values: pd.Series) -> dict[str, Any]:
    shares = values.astype("string").fillna("<NA>").value_counts(normalize=True)
    return {
        "type": "categorical",
        "categories": {str(k): float(s) for k, s in shares.items()},
        "missing_rate": float(values.isna().mean()),
    }


def prediction_profile(p: np.ndarray, threshold: float) -> dict[str, Any]:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, p, side="right"), minlength=len(edges) + 1)
    return {
        "bin_edges": edges.tolist(),
        "proportions": (counts / len(p)).tolist(),
        "mean": float(p.mean()),
        "send_coupon_rate": float((p <= threshold).mean()),
        "n": len(p),
    }


def json_value(v: Any) -> Any:
    if pd.isna(v):
        return None
    return v.item() if hasattr(v, "item") else v


def build_catalogue(paths: Paths) -> list[dict[str, Any]]:
    lines = pd.read_parquet(
        paths.silver_lines, columns=["invoice", "stock_code", "description", "price", "category"]
    )
    g = lines.groupby("stock_code", observed=True)
    cat = pd.DataFrame(
        {
            "description": g["description"].agg(lambda s: str(s.mode().iat[0])),
            "category": g["category"].agg(lambda s: str(s.mode().iat[0])),
            "median_price": g["price"].median().round(2),
            "n_orders": g["invoice"].nunique(),
        }
    ).sort_values("n_orders", ascending=False)
    return [
        {"sku": str(sku), **{str(k): json_value(v) for k, v in row.items()}}
        for sku, row in cat.iterrows()
    ]


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    gold = load_gold(paths)
    train = gold[gold["split"] == "train"]
    val = gold[gold["split"] == "validation"].reset_index(drop=True)
    n_sample = params.monitoring.reference_sample_rows
    demo = val.sample(n=min(N_DEMO_CUSTOMERS, len(val) // 4), random_state=params.seed)
    ref = val.drop(index=demo.index)

    model = joblib.load(paths.model_file)
    p_ref = model.predict_proba(ref[FEATURES])[:, 1]
    rules_eval = read_json(paths.metrics / "rules_evaluation.json")
    features = {
        **{f: numeric_profile(ref[f], n_sample, params.seed) for f in NUMERIC},
        **{f: categorical_profile(ref[f]) for f in CATEGORICAL},
    }
    pooled = {
        **{f: numeric_profile(train[f], n_sample, params.seed) for f in NUMERIC},
        **{f: categorical_profile(train[f]) for f in CATEGORICAL},
    }
    known_shift = sorted(
        (
            feature_drift(f, pooled[f], ref[f], params.monitoring)
            for f in FEATURES
            if f not in params.monitoring.exclude_features
        ),
        key=lambda d: -d["psi"],
    )
    profile = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "source": "data/gold/retention.parquet, calibration snapshot "
        f"{params.windows.cutoffs.validation[0]} minus the {len(demo)} held-out demo customers",
        "n_rows": len(ref),
        "model_version": model.version,
        "features": features,
        "prediction": prediction_profile(p_ref, model.threshold)
        | {"source": "champion on the reference rows"},
        "known_shift_vs_pooled_training": {
            "pooled_training_cutoffs": params.windows.cutoffs.train,
            "share_of_features_psi_red": float(
                np.mean([d["psi"] >= params.monitoring.thresholds.psi.red for d in known_shift])
            ),
            "features": [
                {k: d[k] for k in ("feature", "psi", "ks", "missing_ref", "missing_live", "status")}
                for d in known_shift
            ],
        },
        "recommend": {
            "rule_fire_rate": rules_eval["basket_coverage"],
            "source": "reports/metrics/rules_evaluation.json basket_coverage (temporal holdout)",
        },
    }
    write_json(paths.reference_profile, profile)

    customers = [
        {
            "ref": customer_ref(int(row["customer_id"])),
            "region": str(row["region"]),
            "features": {f: json_value(row[f]) for f in FEATURES},
        }
        for _, row in demo.sort_values("customer_id").iterrows()
    ]
    write_json(
        paths.demo_customers,
        {
            "generated_at": utc_now(),
            "source": f"validation snapshot {params.windows.cutoffs.validation[0]}, "
            f"{len(customers)} customers sampled with seed {params.seed} and held out of the "
            "monitoring reference; pseudonymised refs, no labels",
            "customers": customers,
        },
    )
    catalogue = build_catalogue(paths)
    write_json(paths.catalogue, {"generated_at": utc_now(), "products": catalogue})
    print(
        f"reference profile: {len(features)} features from {len(ref)} rows; "
        f"{len(customers)} demo customers; {len(catalogue)} catalogue products"
    )
    return profile


if __name__ == "__main__":
    run()
