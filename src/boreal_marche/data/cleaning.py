"""Silver layer: documented, ordered cleaning rules with a before/after report.

Each rule is a function ``DataFrame -> DataFrame`` registered with a rationale. The runner records
rows before/after for every rule so the cleaning report (``reports/cleaning_report.md``) is produced
by the same code that cleans the data.

Leakage note: cancellations are *separated*, never netted against earlier invoices. Netting would
delete a pre-cutoff purchase because of a cancellation that happens after the cutoff, which leaks
future information into past feature snapshots. Cancellations are instead kept as a point-in-time
behavioural signal (see ``features.py``).
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.reskin import add_reskin_columns
from boreal_marche.utils import utc_now, write_json

# Duplicate key: description is deliberately excluded — the stock code identifies the product, and
# two scans of one SKU that differ only in spelling ("BUNTING , SPOTTY" vs "SPOTTY BUNTING") are the
# same line. (The first version included description; the data-quality suite caught 2 duplicates
# that survived into silver after R8 canonicalised the spellings.)
SOURCE_KEY = [
    "invoice",
    "stock_code",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]


@dataclass
class RuleResult:
    rule: str
    rows_before: int
    rows_after: int
    rationale: str
    rows_flagged: int = 0

    @property
    def rows_removed(self) -> int:
        return self.rows_before - self.rows_after


def normalise_codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["invoice"] = out["invoice"].str.strip().str.upper()
    out["stock_code"] = out["stock_code"].str.strip().str.upper()
    return out


def drop_sheet_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """The two yearly sheets both contain 2010-12-01..2010-12-09; keep the later sheet's copy."""
    sheets = sorted(df["source_sheet"].unique())
    if len(sheets) < 2:
        return df
    first, second = sheets[0], sheets[1]
    second_invoices = set(df.loc[df["source_sheet"] == second, "invoice"])
    dup = (df["source_sheet"] == first) & df["invoice"].isin(second_invoices)
    return df.loc[~dup]


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=SOURCE_KEY, keep="first")


def is_cancellation(df: pd.DataFrame) -> pd.Series:
    return df["invoice"].str.startswith("C")


def drop_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~is_cancellation(df)]


def drop_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~df["invoice"].str.startswith("A")]


def drop_nonpositive_quantity(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["quantity"] > 0]


def drop_nonpositive_price(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["price"] > 0]


def service_mask(codes: pd.Series, patterns: list[str]) -> pd.Series:
    exact = {p for p in patterns if not p.endswith("*")}
    prefixes = [p for p in patterns if p.endswith("*")]
    uniq = pd.Series(codes.unique())
    hit = uniq.isin(exact) | uniq.map(lambda c: any(fnmatch.fnmatch(c, p) for p in prefixes))
    flagged = set(uniq[hit])
    return codes.isin(flagged)


def canonicalise_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """One canonical description per stock code: the most frequent normalised variant."""
    out = df.copy()
    norm = (
        out["description"].fillna("").str.upper().str.strip().str.replace(r"\s+", " ", regex=True)
    )
    out["description_raw"] = out["description"]
    tmp = pd.DataFrame({"stock_code": out["stock_code"], "d": norm})
    canon = (
        tmp[tmp["d"] != ""]
        .groupby(["stock_code", "d"])
        .size()
        .reset_index(name="n")
        .sort_values(["stock_code", "n", "d"], ascending=[True, False, True])
        .drop_duplicates("stock_code")
        .set_index("stock_code")["d"]
    )
    out["description"] = out["stock_code"].map(canon).fillna("UNKNOWN").astype("string")
    return out


class Cleaner:
    """Runs the ordered rules and records their effect."""

    def __init__(self) -> None:
        self.results: list[RuleResult] = []

    def apply(
        self,
        df: pd.DataFrame,
        name: str,
        fn: Callable[[pd.DataFrame], pd.DataFrame],
        rationale: str,
    ) -> pd.DataFrame:
        before = len(df)
        out = fn(df)
        self.results.append(RuleResult(name, before, len(out), rationale))
        print(f"{name:<34} {before:>10,} -> {len(out):>10,}  (-{before - len(out):,})")
        return out


def clean(
    bronze: pd.DataFrame, params: Params
) -> tuple[pd.DataFrame, pd.DataFrame, list[RuleResult], dict[str, object]]:
    c = Cleaner()
    df = c.apply(
        bronze,
        "R0 normalise codes",
        normalise_codes,
        "Stock codes/invoices contain trailing spaces and mixed case ('47503J ', 'm'); "
        "without normalisation the same SKU is counted twice.",
    )
    df = c.apply(
        df,
        "R1 drop sheet overlap",
        drop_sheet_overlap,
        "Both yearly sheets contain 2010-12-01..2010-12-09. Naive concatenation "
        "double-counts those orders, inflating frequency and monetary features.",
    )
    df = c.apply(
        df,
        "R2 drop exact duplicate lines",
        drop_exact_duplicates,
        "Identical lines (same invoice, SKU, quantity, timestamp, price, customer) are "
        "system double-scans; they inflate quantities and basket co-occurrence. "
        "Description is not part of the key (spelling variants of one SKU).",
    )
    cancellations = df.loc[is_cancellation(df)].copy()
    df = c.apply(
        df,
        "R3 separate cancellations",
        drop_cancellations,
        "Invoices prefixed 'C' are reversals, not purchases. Kept in a separate table used "
        "only as a point-in-time feature; never netted (netting leaks future events).",
    )
    df = c.apply(
        df,
        "R4 drop bad-debt adjustments",
        drop_adjustments,
        "Invoices prefixed 'A' are accounting adjustments ('Adjust bad debt').",
    )
    df = c.apply(
        df,
        "R5 drop non-positive quantity",
        drop_nonpositive_quantity,
        "Non-cancellation lines with quantity <= 0 are stock write-offs/damages "
        "('lost', 'damaged', '?') with no customer.",
    )
    df = c.apply(
        df,
        "R6 drop non-positive price",
        drop_nonpositive_price,
        "Price <= 0 lines are free samples or manual corrections; they are not sales and "
        "would distort monetary features.",
    )
    patterns = params.cleaning.service_stock_codes
    df = c.apply(
        df,
        "R7 drop service / fee SKUs",
        lambda d: d.loc[~service_mask(d["stock_code"], patterns)],
        "Postage, carriage, manual, bank/Amazon fees, test and gift-voucher codes are not "
        "products: they would dominate baskets ('recommend postage') and skew category mix.",
    )
    before_desc = df["description"].fillna("").str.upper().str.strip().nunique()
    df = c.apply(
        df,
        "R8 canonicalise descriptions",
        canonicalise_descriptions,
        "The same SKU appears under several spellings; one canonical label per stock code "
        "keeps the rule engine and catalogue consistent.",
    )
    desc_changed = int(
        (
            df["description_raw"]
            .fillna("")
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            != df["description"]
        ).sum()
    )
    c.results[-1].rows_flagged = desc_changed

    # R9: outlier flag + cap. Threshold estimated on the training period only (before the first
    # validation cutoff) so that no statistic of future data enters past features.
    df = df.copy()
    df["line_revenue"] = df["quantity"].astype("float64") * df["price"]
    ref_end = pd.Timestamp(params.windows.cutoffs.validation[0])
    ref = df.loc[df["invoice_date"] < ref_end]
    q = params.cleaning.outlier_quantile
    qty_cap = float(ref["quantity"].quantile(q))
    rev_cap = float(ref["line_revenue"].quantile(q))
    df["is_outlier"] = (df["quantity"] > qty_cap) | (df["line_revenue"] > rev_cap)
    df["line_revenue_capped"] = df["line_revenue"].clip(upper=rev_cap)
    c.results.append(
        RuleResult(
            "R9 flag + cap outliers",
            len(df),
            len(df),
            f"Lines above the {q:.1%} quantile of quantity ({qty_cap:,.0f}) or line revenue "
            f"({rev_cap:,.2f}) — e.g. the 80,995-unit error — are flagged and revenue is "
            "capped for "
            "features instead of dropped (they are real orders but not representative). Caps are "
            "estimated on pre-validation data only.",
            rows_flagged=int(df["is_outlier"].sum()),
        )
    )

    df["has_customer"] = df["customer_id"].notna()
    df = add_reskin_columns(df)
    cancellations = add_reskin_columns(cancellations)
    no_cust = int((~df["has_customer"]).sum())
    c.results.append(
        RuleResult(
            "R10 null customer id (Head A view)",
            len(df),
            len(df) - no_cust,
            "Guest checkouts cannot be attributed to a customer, so they are excluded from the "
            "retention table (Head A) but kept for basket mining (Head B) where the invoice is "
            "the unit.",
            rows_flagged=no_cust,
        )
    )
    extras: dict[str, object] = {
        "qty_cap": qty_cap,
        "rev_cap": rev_cap,
        "distinct_descriptions_before": int(before_desc),
        "distinct_descriptions_after": int(df["description"].nunique()),
    }
    return df.reset_index(drop=True), cancellations.reset_index(drop=True), c.results, extras


def summarise(df: pd.DataFrame) -> dict[str, object]:
    head_a = df[df["has_customer"]]
    return {
        "lines": int(len(df)),
        "invoices": int(df["invoice"].nunique()),
        "skus": int(df["stock_code"].nunique()),
        "customers": int(head_a["customer_id"].nunique()),
        "head_a_lines": int(len(head_a)),
        "head_a_invoices": int(head_a["invoice"].nunique()),
        "head_b_invoices": int(df["invoice"].nunique()),
        "revenue": float(df["line_revenue"].sum()),
        "date_min": str(df["invoice_date"].min()),
        "date_max": str(df["invoice_date"].max()),
        "mean_lines_per_invoice": float(df.groupby("invoice").size().mean()),
    }


def report_markdown(
    results: list[RuleResult],
    before: dict[str, object],
    after: dict[str, object],
    extras: dict[str, object],
) -> str:
    lines = [
        "# Cleaning report",
        "",
        f"Generated {utc_now()} by `make etl` (`boreal_marche.data.cleaning`).",
        "",
        "| Rule | Rows before | Rows after | Removed | Flagged | Rationale |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.rule} | {r.rows_before:,} | {r.rows_after:,} | {r.rows_removed:,} | "
            f"{r.rows_flagged:,} | {r.rationale} |"
        )
    lines += ["", "## Before / after", "", "| Measure | Bronze | Silver |", "|---|---:|---:|"]

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}"
        return f"{v:,}" if isinstance(v, int) else str(v)

    for k in after:
        b, a = before.get(k, ""), after[k]
        lines.append(f"| {k} | {fmt(b)} | {fmt(a)} |")
    lines += [
        "",
        f"Distinct descriptions: {extras['distinct_descriptions_before']:,} -> "
        f"{extras['distinct_descriptions_after']:,}.",
        "",
    ]
    return "\n".join(lines)


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, object]:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    bronze = pd.read_parquet(paths.bronze_transactions)
    bronze_summary = {
        "lines": int(len(bronze)),
        "invoices": int(bronze["invoice"].nunique()),
        "skus": int(bronze["stock_code"].nunique()),
        "customers": int(bronze["customer_id"].nunique()),
        "revenue": float((bronze["quantity"] * bronze["price"]).sum()),
        "date_min": str(bronze["invoice_date"].min()),
        "date_max": str(bronze["invoice_date"].max()),
    }
    silver, cancellations, results, extras = clean(bronze, params)
    silver.to_parquet(paths.silver_lines, index=False)
    cancellations.to_parquet(paths.silver_cancellations, index=False)
    after = summarise(silver)
    payload = {
        "generated_at": utc_now(),
        "rules": [{**r.__dict__, "rows_removed": r.rows_removed} for r in results],
        "bronze": bronze_summary,
        "silver": after,
        "extras": extras,
        "cancellation_lines": int(len(cancellations)),
    }
    write_json(paths.metrics / "cleaning_report.json", payload)
    (paths.reports / "cleaning_report.md").write_text(
        report_markdown(results, bronze_summary, after, extras), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    run()
