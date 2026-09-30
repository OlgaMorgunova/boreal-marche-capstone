"""Gold layer: point-in-time customer feature snapshots for Head A (retention).

For a cutoff ``T`` (declared in ``params.yaml``):

* eligible customers: at least one valid invoice in the observation window ``[T - obs, T)``;
* features: computed only from rows with ``invoice_date < T`` (windowed features use
  ``[T - obs, T)``, lifetime features use all history ``< T``);
* label: ``1`` if the customer has at least one valid invoice in ``[T, T + label)``.

``build_snapshot(..., feature_end_offset_days=label)`` deliberately computes features *through* the
label window; it exists only to demonstrate how much a leak inflates the metric (D3 leakage section)
and is never used for training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.reskin import REGIONS
from boreal_marche.utils import utc_now, write_json

DATA_START = pd.Timestamp("2009-12-01")
MIX_CATEGORIES = ["seasonal", "kitchen", "home_decor", "bags_wrap", "lighting"]


@dataclass(frozen=True)
class FeatureDef:
    name: str
    family: str
    dtype: str
    definition: str
    rationale: str
    missing: str = "never missing (0 is a true zero)"


_NA_GAP = (
    "NaN when fewer than 2 distinct order days (gap undefined); imputed with median + indicator"
)
FEATURE_SPEC: list[FeatureDef] = [
    FeatureDef(
        "recency_days",
        "recency",
        "float",
        "Days from last order to cutoff",
        "Strongest classical churn signal (RFM).",
    ),
    FeatureDef(
        "tenure_days",
        "recency",
        "float",
        "Days from first order in history to cutoff",
        "Long-standing customers churn less.",
        "never missing; left-censored at 2009-12-01 (see tenure_censored)",
    ),
    FeatureDef(
        "tenure_censored",
        "recency",
        "int",
        "1 if first observed order is within 30 days of the data start (true tenure unknown)",
        "Makes left-censoring explicit instead of pretending tenure is exact.",
    ),
    FeatureDef(
        "orders_30d",
        "frequency",
        "int",
        "Distinct invoices in [T-30d, T)",
        "Very recent engagement.",
    ),
    FeatureDef(
        "orders_90d",
        "frequency",
        "int",
        "Distinct invoices in [T-90d, T)",
        "Same horizon as the label window.",
    ),
    FeatureDef(
        "orders_180d",
        "frequency",
        "int",
        "Distinct invoices in the observation window",
        "Core frequency signal.",
    ),
    FeatureDef(
        "orders_lifetime",
        "frequency",
        "int",
        "Distinct invoices before T",
        "Habit strength; left-censored.",
    ),
    FeatureDef(
        "active_months_180d",
        "frequency",
        "int",
        "Distinct calendar months with an order in the observation window",
        "Regularity vs. one burst of orders.",
    ),
    FeatureDef("revenue_30d", "monetary", "float", "Capped revenue in [T-30d, T)", "Recent spend."),
    FeatureDef("revenue_90d", "monetary", "float", "Capped revenue in [T-90d, T)", "Recent spend."),
    FeatureDef(
        "revenue_180d",
        "monetary",
        "float",
        "Capped revenue in the observation window",
        "Customer value; also the spend-quartile for the fairness audit.",
    ),
    FeatureDef(
        "revenue_lifetime",
        "monetary",
        "float",
        "Capped revenue before T",
        "Lifetime value proxy; left-censored.",
    ),
    FeatureDef(
        "aov_180d",
        "monetary",
        "float",
        "Mean order value in the observation window",
        "Wholesale vs. small buyers behave differently.",
    ),
    FeatureDef(
        "max_order_value_180d",
        "monetary",
        "float",
        "Largest order value in window",
        "Captures one-off bulk orders.",
    ),
    FeatureDef(
        "mean_unit_price_180d",
        "monetary",
        "float",
        "Mean unit price of lines in window",
        "Price tier of the assortment bought.",
    ),
    FeatureDef(
        "mean_gap_days",
        "cadence",
        "float",
        "Mean days between consecutive distinct order days (lifetime)",
        "Expected re-order cycle.",
        _NA_GAP,
    ),
    FeatureDef(
        "std_gap_days",
        "cadence",
        "float",
        "Std of days between distinct order days (lifetime)",
        "Regular vs. erratic buyers.",
        "NaN when fewer than 3 order days; median + indicator",
    ),
    FeatureDef(
        "overdue_ratio",
        "cadence",
        "float",
        "recency_days / mean_gap_days",
        ">1 means the customer is late versus their own cycle.",
        _NA_GAP,
    ),
    FeatureDef(
        "orders_trend",
        "cadence",
        "float",
        "orders in last 90d minus orders in the previous 90d",
        "Accelerating vs. decelerating engagement.",
    ),
    FeatureDef(
        "revenue_trend",
        "cadence",
        "float",
        "(revenue last 90d - revenue previous 90d) / revenue_180d",
        "Scale-free spend trend in [-1, 1].",
    ),
    FeatureDef(
        "lines_per_order_180d", "basket", "float", "Mean invoice lines per order", "Basket breadth."
    ),
    FeatureDef(
        "units_per_line_180d",
        "basket",
        "float",
        "Mean units per line",
        "Wholesale (large multiples) vs. retail buying.",
    ),
    FeatureDef(
        "distinct_skus_180d",
        "basket",
        "int",
        "Distinct stock codes bought in window",
        "Assortment breadth.",
    ),
    FeatureDef(
        "sku_repeat_rate_180d",
        "basket",
        "float",
        "1 - distinct SKUs / lines in window",
        "Replenishment buyers repeat SKUs.",
    ),
    *[
        FeatureDef(
            f"share_{c}",
            "category",
            "float",
            f"Share of window revenue in category '{c}' (keyword-derived)",
            "Category mix; seasonal buyers have predictable gaps.",
        )
        for c in MIX_CATEGORIES
    ],
    FeatureDef(
        "category_entropy_180d",
        "category",
        "float",
        "Shannon entropy of category revenue shares",
        "Diversified buyers are stickier.",
    ),
    FeatureDef(
        "cancel_lines_180d",
        "behaviour",
        "int",
        "Cancellation lines in window (point-in-time, from the separated cancellation table)",
        "Dissatisfaction signal.",
    ),
    FeatureDef(
        "cancel_rate_180d",
        "behaviour",
        "float",
        "cancel lines / (purchase + cancel lines)",
        "Normalised dissatisfaction.",
    ),
    FeatureDef(
        "weekend_share_180d",
        "behaviour",
        "float",
        "Share of orders placed on Sat/Sun",
        "Business vs. consumer buying pattern.",
    ),
    FeatureDef(
        "mean_order_hour_180d",
        "behaviour",
        "float",
        "Mean hour of day of orders",
        "Business-hours buyers are likely B2B.",
    ),
    FeatureDef(
        "outlier_line_share_180d",
        "behaviour",
        "float",
        "Share of lines flagged as outliers by cleaning rule R9",
        "Keeps information about bulk orders that were capped.",
    ),
    FeatureDef(
        "region",
        "geography",
        "category",
        "Re-skinned delivery region of the last order (QC/ON/MARITIMES/INTL)",
        "Delivery geography; also a fairness-audit subgroup.",
        "never missing; unknown countries map to INTL",
    ),
    FeatureDef(
        "season_q",
        "context",
        "category",
        "Calendar quarter of the cutoff (1-4)",
        "Label base-rate is seasonal (Q4 peak); known at prediction time.",
        "never missing (derived from the cutoff date)",
    ),
]
FEATURES: list[str] = [f.name for f in FEATURE_SPEC]
CATEGORICAL: list[str] = [f.name for f in FEATURE_SPEC if f.dtype == "category"]
NUMERIC: list[str] = [f.name for f in FEATURE_SPEC if f.dtype != "category"]
ID_COLS = ["customer_id", "cutoff"]
TARGET = "y"
# Columns that exist in the gold table for evaluation but must never be used as features.
NON_FEATURE_COLS = ["future_revenue_90d", "split"]


def _invoices(lines: pd.DataFrame) -> pd.DataFrame:
    g = lines.groupby("invoice", sort=False)
    inv = g.agg(
        customer_id=("customer_id", "first"),
        invoice_date=("invoice_date", "min"),
        revenue=("line_revenue_capped", "sum"),
        n_lines=("stock_code", "size"),
        units=("quantity", "sum"),
        region=("region", "last"),
    )
    return inv.reset_index()


def build_snapshot(
    lines: pd.DataFrame,
    cancellations: pd.DataFrame,
    cutoff: pd.Timestamp,
    obs_days: int,
    label_days: int,
    feature_end_offset_days: int = 0,
) -> pd.DataFrame:
    """Features + label for one cutoff. ``feature_end_offset_days > 0`` creates a LEAKY snapshot."""
    lines = lines.loc[lines["has_customer"]]
    feat_end = cutoff + pd.Timedelta(days=feature_end_offset_days)
    obs_start = cutoff - pd.Timedelta(days=obs_days)
    hist = lines.loc[lines["invoice_date"] < feat_end]
    win = hist.loc[hist["invoice_date"] >= obs_start]
    customers = pd.Index(win["customer_id"].unique(), name="customer_id")
    hist = hist.loc[hist["customer_id"].isin(customers)]
    win = win.loc[win["customer_id"].isin(customers)]

    inv_hist = _invoices(hist)
    inv_win = inv_hist.loc[inv_hist["invoice_date"] >= obs_start]
    f = pd.DataFrame(index=customers)
    days = lambda s: s.dt.total_seconds() / 86400.0  # noqa: E731

    gh = inv_hist.groupby("customer_id")
    last = gh["invoice_date"].max()
    first = gh["invoice_date"].min()
    f["recency_days"] = days(feat_end - last)
    f["tenure_days"] = days(feat_end - first)
    f["tenure_censored"] = (first < DATA_START + pd.Timedelta(days=30)).astype(int)
    for d in (30, 90):
        recent = inv_win.loc[inv_win["invoice_date"] >= feat_end - pd.Timedelta(days=d)]
        f[f"orders_{d}d"] = recent.groupby("customer_id").size()
        f[f"revenue_{d}d"] = recent.groupby("customer_id")["revenue"].sum()
    gw = inv_win.groupby("customer_id")
    f["orders_180d"] = gw.size()
    f["orders_lifetime"] = gh.size()
    f["active_months_180d"] = gw["invoice_date"].agg(lambda s: s.dt.to_period("M").nunique())
    f["revenue_180d"] = gw["revenue"].sum()
    f["revenue_lifetime"] = gh["revenue"].sum()
    f["aov_180d"] = gw["revenue"].mean()
    f["max_order_value_180d"] = gw["revenue"].max()
    f["mean_unit_price_180d"] = win.groupby("customer_id")["price"].mean()

    # Gaps between distinct order *days*: several invoices on the same day are one visit, otherwise
    # zero-day gaps make the cadence (and overdue_ratio) explode.
    order_days = (
        inv_hist.assign(day=inv_hist["invoice_date"].dt.normalize())[["customer_id", "day"]]
        .drop_duplicates()
        .sort_values(["customer_id", "day"])
    )
    gaps = days(order_days.groupby("customer_id")["day"].diff()).dropna()
    gap_df = pd.DataFrame({"customer_id": order_days.loc[gaps.index, "customer_id"], "gap": gaps})
    f["mean_gap_days"] = gap_df.groupby("customer_id")["gap"].mean()
    f["std_gap_days"] = gap_df.groupby("customer_id")["gap"].std()
    f["overdue_ratio"] = f["recency_days"] / f["mean_gap_days"].replace(0, np.nan)

    mid = feat_end - pd.Timedelta(days=obs_days / 2)
    last_half = inv_win.loc[inv_win["invoice_date"] >= mid].groupby("customer_id")
    prev_half = inv_win.loc[inv_win["invoice_date"] < mid].groupby("customer_id")
    o_last = last_half.size().reindex(customers, fill_value=0)
    o_prev = prev_half.size().reindex(customers, fill_value=0)
    r_last = last_half["revenue"].sum().reindex(customers, fill_value=0.0)
    r_prev = prev_half["revenue"].sum().reindex(customers, fill_value=0.0)
    f["orders_trend"] = (o_last - o_prev).astype(float)
    f["revenue_trend"] = (r_last - r_prev) / (r_last + r_prev).replace(0, np.nan)

    f["lines_per_order_180d"] = gw["n_lines"].mean()
    f["units_per_line_180d"] = win.groupby("customer_id")["quantity"].mean()
    f["distinct_skus_180d"] = win.groupby("customer_id")["stock_code"].nunique()
    f["sku_repeat_rate_180d"] = 1 - f["distinct_skus_180d"] / win.groupby("customer_id").size()

    cat_rev = win.pivot_table(
        index="customer_id",
        columns="category",
        values="line_revenue_capped",
        aggfunc="sum",
        fill_value=0.0,
    )
    shares = cat_rev.div(cat_rev.sum(axis=1).replace(0, np.nan), axis=0)
    for c in MIX_CATEGORIES:
        f[f"share_{c}"] = shares.get(c, 0.0)
    p = shares.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    f["category_entropy_180d"] = pd.Series(ent, index=shares.index)

    canc = cancellations.loc[
        (cancellations["invoice_date"] >= obs_start)
        & (cancellations["invoice_date"] < feat_end)
        & cancellations["customer_id"].isin(customers)
    ]
    f["cancel_lines_180d"] = canc.groupby("customer_id").size()
    n_lines = win.groupby("customer_id").size()
    f["cancel_rate_180d"] = f["cancel_lines_180d"].fillna(0) / (
        n_lines + f["cancel_lines_180d"].fillna(0)
    )
    f["weekend_share_180d"] = gw["invoice_date"].agg(lambda s: (s.dt.dayofweek >= 5).mean())
    f["mean_order_hour_180d"] = gw["invoice_date"].agg(lambda s: s.dt.hour.mean())
    f["outlier_line_share_180d"] = win.groupby("customer_id")["is_outlier"].mean()
    last_inv = inv_win.sort_values("invoice_date").groupby("customer_id").tail(1)
    f["region"] = last_inv.set_index("customer_id")["region"]
    f["season_q"] = str(cutoff.quarter)

    count_cols = ["orders_30d", "orders_90d", "revenue_30d", "revenue_90d", "cancel_lines_180d"]
    f[count_cols] = f[count_cols].fillna(0)

    label_end = cutoff + pd.Timedelta(days=label_days)
    fut = lines.loc[
        (lines["invoice_date"] >= cutoff)
        & (lines["invoice_date"] < label_end)
        & lines["customer_id"].isin(customers)
    ]
    f[TARGET] = customers.isin(fut["customer_id"].unique()).astype(int)
    f["future_revenue_90d"] = fut.groupby("customer_id")["line_revenue_capped"].sum()
    f["future_revenue_90d"] = f["future_revenue_90d"].fillna(0.0)
    f["cutoff"] = cutoff
    f = f.reset_index()
    for col in CATEGORICAL:
        f[col] = f[col].astype("string")
    int_cols = [x.name for x in FEATURE_SPEC if x.dtype == "int"]
    f[int_cols] = f[int_cols].astype("int64")
    return f[ID_COLS + FEATURES + [TARGET, "future_revenue_90d"]]


def build_all(
    params: Params,
    lines: pd.DataFrame,
    cancellations: pd.DataFrame,
    feature_end_offset_days: int = 0,
) -> pd.DataFrame:
    w = params.windows
    frames = []
    for split, cutoffs in (
        ("train", w.cutoffs.train),
        ("validation", w.cutoffs.validation),
        ("test", w.cutoffs.test),
    ):
        for c in cutoffs:
            snap = build_snapshot(
                lines,
                cancellations,
                pd.Timestamp(c),
                w.observation_days,
                w.label_days,
                feature_end_offset_days,
            )
            snap["split"] = split
            frames.append(snap)
    return pd.concat(frames, ignore_index=True)


def spec_markdown() -> str:
    rows = [
        "# Feature specification (Head A)",
        "",
        f"{len(FEATURES)} features. Source: `boreal_marche.features.FEATURE_SPEC`. All features "
        "are computed from rows strictly before the cutoff `T`.",
        "",
        "| # | Feature | Family | Type | Definition | Rationale | Missingness policy |",
        "|---:|---|---|---|---|---|---|",
    ]
    for i, f in enumerate(FEATURE_SPEC, 1):
        rows.append(
            f"| {i} | `{f.name}` | {f.family} | {f.dtype} | {f.definition} | "
            f"{f.rationale} | {f.missing} |"
        )
    return "\n".join(rows) + "\n"


def run(params: Params | None = None, paths: Paths | None = None) -> pd.DataFrame:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    lines = pd.read_parquet(paths.silver_lines)
    canc = pd.read_parquet(paths.silver_cancellations)
    gold = build_all(params, lines, canc)
    gold.to_parquet(paths.gold / "retention.parquet", index=False)
    summary = (
        gold.groupby(["split", "cutoff"])
        .agg(customers=("customer_id", "size"), positive_rate=(TARGET, "mean"))
        .reset_index()
    )
    print(summary.to_string(index=False))
    write_json(
        paths.metrics / "gold_summary.json",
        {
            "generated_at": utc_now(),
            "n_features": len(FEATURES),
            "features": FEATURES,
            "regions": REGIONS,
            "snapshots": summary.assign(cutoff=summary["cutoff"].astype(str)).to_dict("records"),
            "missing_share": gold[FEATURES].isna().mean().round(4).to_dict(),
        },
    )
    (paths.reports / "feature_spec.md").write_text(spec_markdown(), encoding="utf-8")
    return gold


if __name__ == "__main__":
    run()
