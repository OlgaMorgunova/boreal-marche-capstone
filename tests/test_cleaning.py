"""Cleaning rules on a hand-made bronze frame: each rule removes exactly what it claims."""

from __future__ import annotations

import pandas as pd

from boreal_marche.data.cleaning import clean, service_mask
from boreal_marche.data.quality import check_silver


def bronze() -> pd.DataFrame:
    rows = [
        # invoice, stock, desc, qty, date, price, cust, country, sheet
        ("500001", "85123A", "WHITE HEART", 6, "2010-01-05 10:00", 2.5, 1, "United Kingdom", "Y1"),
        ("500001", "85123A", "WHITE HEART", 6, "2010-01-05 10:00", 2.5, 1, "United Kingdom", "Y1"),
        ("500002", "47503J ", "Bag ", 1, "2010-02-01 10:00", 1.0, 2, "France", "Y1"),
        (
            "C500003",
            "85123A",
            "WHITE HEART",
            -2,
            "2010-02-02 10:00",
            2.5,
            1,
            "United Kingdom",
            "Y1",
        ),
        (
            "A500004",
            "B",
            "Adjust bad debt",
            1,
            "2010-02-03 10:00",
            100.0,
            None,
            "United Kingdom",
            "Y1",
        ),
        ("500005", "22000", "LOST", -5, "2010-02-04 10:00", 0.0, None, "United Kingdom", "Y1"),
        ("500006", "22001", "FREE", 1, "2010-02-05 10:00", 0.0, 3, "EIRE", "Y1"),
        ("500007", "POST", "POSTAGE", 1, "2010-02-06 10:00", 18.0, 3, "EIRE", "Y1"),
        ("500008", "85123A", "white  heart", 2, "2010-12-02 10:00", 2.5, 4, "Japan", "Y1"),
        ("500008", "85123A", "white  heart", 2, "2010-12-02 10:00", 2.5, 4, "Japan", "Y2"),
        ("500009", "85123A", "CREAM HEART", 1, "2010-12-05 10:00", 2.5, None, "Unspecified", "Y2"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "invoice",
            "stock_code",
            "description",
            "quantity",
            "invoice_date",
            "price",
            "customer_id",
            "country",
            "source_sheet",
        ],
    )
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype("Int64")
    for c in ("invoice", "stock_code", "description", "country"):
        df[c] = df[c].astype("string")
    df.insert(0, "row_id", range(len(df)))
    return df


def test_rules_remove_expected_rows(params) -> None:
    silver, canc, results, _ = clean(bronze(), params)
    removed = {r.rule.split()[0]: r.rows_removed for r in results}
    assert removed["R1"] == 1  # sheet overlap copy of invoice 500008
    assert removed["R2"] == 1  # exact duplicate of 500001
    assert removed["R3"] == 1 and len(canc) == 1
    assert removed["R4"] == 1
    assert removed["R5"] == 1
    assert removed["R6"] == 1
    assert removed["R7"] == 1  # POST
    assert set(silver["invoice"]) == {"500001", "500002", "500008", "500009"}
    assert "47503J" in set(silver["stock_code"])  # trailing space normalised


def test_descriptions_canonical_and_guests_kept(params) -> None:
    silver, _, _, _ = clean(bronze(), params)
    assert silver.loc[silver["stock_code"] == "85123A", "description"].nunique() == 1
    assert (~silver["has_customer"]).sum() == 1  # guest kept for Head B
    assert set(silver["region"]) <= {"QC", "ON", "MARITIMES", "INTL"}


def test_silver_passes_quality_suite(params) -> None:
    silver, _, _, _ = clean(bronze(), params)
    assert check_silver(silver, params.cleaning.service_stock_codes) == []


def test_service_patterns() -> None:
    codes = pd.Series(["POST", "ADJUST2", "TEST001", "GIFT_0001_20", "85123A", "DCGS0058"])
    assert service_mask(codes, ["POST", "ADJUST*", "TEST*", "GIFT_0001_*"]).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
