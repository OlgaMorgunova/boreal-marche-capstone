"""Reproducible ingestion: local acquisition, checksum verification, explicit schema, manifest."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from boreal_marche.config import Paths, load_params
from boreal_marche.data import ingest
from boreal_marche.utils import sha256_file


def write_xlsx(path) -> None:  # type: ignore[no-untyped-def]
    df = pd.DataFrame(
        {
            "Invoice": ["489434", "C489449"],
            "StockCode": ["85048", "22087"],
            "Description": ["LED CANDLE", "PAPER BUNTING"],
            "Quantity": [12, -12],
            "InvoiceDate": pd.to_datetime(["2009-12-01 07:45", "2009-12-01 10:33"]),
            "Price": [6.95, 2.95],
            "Customer ID": [13085.0, None],
            "Country": ["United Kingdom", "Australia"],
        }
    )
    with pd.ExcelWriter(path) as xw:
        df.to_excel(xw, sheet_name="Year 2009-2010", index=False)
        df.to_excel(xw, sheet_name="Year 2010-2011", index=False)


def test_ingest_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "doc" / "online_retail_II.xlsx"
    src.parent.mkdir()
    write_xlsx(src)
    base = load_params()
    params = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "expected_sha256": sha256_file(src),
                    "local_candidates": ["doc/online_retail_II.xlsx"],
                }
            )
        }
    )
    paths = Paths(tmp_path)
    manifest = ingest.run(params, paths)
    assert manifest["row_count"] == 4
    assert manifest["sheet_row_counts"] == {"Year 2009-2010": 2, "Year 2010-2011": 2}
    assert json.loads(paths.manifest.read_text())["sha256"] == sha256_file(src)
    schema = pq.read_schema(paths.bronze_transactions)
    assert schema.field("customer_id").type == "int64"
    assert schema.field("quantity").type == "int32"
    bronze = pd.read_parquet(paths.bronze_transactions)
    assert bronze["customer_id"].isna().sum() == 2
    # idempotent second run uses the existing data/raw copy
    assert ingest.run(params, paths)["acquisition"] == "existing data/raw copy"


def test_tampered_file_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    src = tmp_path / "doc" / "online_retail_II.xlsx"
    src.parent.mkdir()
    write_xlsx(src)
    base = load_params()
    params = base.model_copy(
        update={
            "data": base.data.model_copy(update={"local_candidates": ["doc/online_retail_II.xlsx"]})
        }
    )
    with pytest.raises(ingest.ChecksumError):
        ingest.run(params, Paths(tmp_path))
