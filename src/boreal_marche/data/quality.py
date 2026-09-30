"""Data-quality assertion suite for the silver and gold layers (run in CI and by ``make dq``).

Each check returns a list of human-readable failures; an empty list means the layer is valid.
"""

from __future__ import annotations

import sys

import pandas as pd

from boreal_marche.config import Paths, load_params
from boreal_marche.data.cleaning import SOURCE_KEY, service_mask
from boreal_marche.features import FEATURES, NON_FEATURE_COLS, TARGET
from boreal_marche.utils import read_json


def check_silver(df: pd.DataFrame, service_patterns: list[str]) -> list[str]:
    f: list[str] = []
    if (df["quantity"] <= 0).any():
        f.append("non-positive quantity present")
    if (df["price"] <= 0).any():
        f.append("non-positive price present")
    if df["invoice"].str.startswith(("C", "A")).any():
        f.append("cancellation/adjustment invoices present")
    if service_mask(df["stock_code"], service_patterns).any():
        f.append("service SKUs present")
    if df.duplicated(subset=SOURCE_KEY).any():
        f.append("exact duplicate lines present")
    if (df["stock_code"] != df["stock_code"].str.strip().str.upper()).any():
        f.append("stock codes not normalised")
    if df.groupby("stock_code")["description"].nunique().max() > 1:
        f.append("stock code with more than one canonical description")
    if df["invoice_date"].isna().any():
        f.append("null invoice dates")
    if not df["region"].isin(["QC", "ON", "MARITIMES", "INTL"]).all():
        f.append("unknown region label")
    return f


def check_gold(df: pd.DataFrame) -> list[str]:
    f: list[str] = []
    missing = [c for c in FEATURES + [TARGET, "customer_id", "cutoff"] if c not in df.columns]
    if missing:
        f.append(f"missing columns {missing}")
        return f
    if leaked := [c for c in NON_FEATURE_COLS if c in FEATURES]:
        f.append(f"evaluation-only columns used as features: {leaked}")
    if df.duplicated(["customer_id", "cutoff"]).any():
        f.append("duplicate (customer, cutoff) snapshot rows")
    if not df[TARGET].isin([0, 1]).all():
        f.append("target not binary")
    if (df["recency_days"] < 0).any():
        f.append("negative recency: a feature row saw data after its cutoff")
    obs = load_params().windows.observation_days
    if (df["recency_days"] > obs).any():
        f.append("recency beyond the observation window: eligibility rule broken")
    no_na = [
        c
        for c in FEATURES
        if c not in ("mean_gap_days", "std_gap_days", "overdue_ratio", "revenue_trend")
    ]
    if nulls := [c for c in no_na if df[c].isna().any()]:
        f.append(f"unexpected nulls in {nulls}")
    rate = df.groupby("cutoff")[TARGET].mean()
    if ((rate < 0.1) | (rate > 0.9)).any():
        f.append(f"implausible label rate per cutoff: {rate.round(3).to_dict()}")
    return f


def main() -> int:
    params, paths = load_params(), Paths()
    failures: dict[str, list[str]] = {}
    manifest = read_json(paths.manifest)
    if manifest.get("sha256") != params.data.expected_sha256:
        failures["manifest"] = ["manifest checksum does not match params.yaml"]
    failures["silver"] = check_silver(
        pd.read_parquet(paths.silver_lines), params.cleaning.service_stock_codes
    )
    failures["gold"] = check_gold(pd.read_parquet(paths.gold / "retention.parquet"))
    bad = {k: v for k, v in failures.items() if v}
    for layer, items in failures.items():
        print(f"{layer:<9} {'OK' if not items else 'FAIL: ' + '; '.join(items)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
