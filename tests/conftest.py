"""Shared fixtures: small synthetic data so unit tests run in CI without the real dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boreal_marche.config import Paths, load_params

REPO = Path(__file__).resolve().parents[1]


def has_real_data() -> bool:
    return (REPO / "data" / "silver" / "lines.parquet").exists()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if has_real_data():
        return
    skip = pytest.mark.skip(reason="real dataset not present (run `make data etl features`)")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def params():  # type: ignore[no-untyped-def]
    return load_params()


@pytest.fixture(scope="session")
def repo_paths() -> Paths:
    return Paths(REPO)


def make_lines(rows: list[tuple[str, int | None, str, str, int, float]]) -> pd.DataFrame:
    """rows: (invoice, customer_id, stock_code, iso timestamp, quantity, price) -> silver-like."""
    df = pd.DataFrame(
        rows, columns=["invoice", "customer_id", "stock_code", "invoice_date", "quantity", "price"]
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("Int64")
    df["description"] = "ITEM " + df["stock_code"]
    df["line_revenue"] = df["quantity"] * df["price"]
    df["line_revenue_capped"] = df["line_revenue"]
    df["is_outlier"] = False
    df["has_customer"] = df["customer_id"].notna()
    df["region"] = "QC"
    df["category"] = "kitchen"
    df["country"] = "United Kingdom"
    return df


@pytest.fixture()
def synthetic_lines() -> pd.DataFrame:
    """Customer 1 buys before and after the cutoff (2011-01-01); customer 2 only before."""
    return make_lines(
        [
            ("1001", 1, "A", "2010-09-10 10:00", 2, 5.0),
            ("1001", 1, "B", "2010-09-10 10:00", 1, 3.0),
            ("1002", 1, "A", "2010-11-20 12:00", 4, 5.0),
            ("1003", 2, "C", "2010-12-15 09:00", 10, 1.0),
            ("1004", 1, "B", "2011-02-01 11:00", 1, 3.0),  # inside the label window
            ("1005", 2, "C", "2011-06-01 11:00", 1, 1.0),  # after the label window
            ("1006", None, "A", "2010-12-20 15:00", 1, 5.0),  # guest checkout
        ]
    )


@pytest.fixture()
def empty_cancellations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.Series([], dtype="Int64"),
            "invoice_date": pd.Series([], dtype="datetime64[ns]"),
            "stock_code": pd.Series([], dtype="string"),
        }
    )


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
