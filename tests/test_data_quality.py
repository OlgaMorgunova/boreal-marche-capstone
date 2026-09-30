"""Data-quality suite on the real pipeline outputs (skipped when the dataset is absent)."""

from __future__ import annotations

import pandas as pd
import pytest

from boreal_marche.data.ingest import ChecksumError, verify
from boreal_marche.data.quality import check_gold, check_silver
from boreal_marche.utils import read_json


def test_checksum_mismatch_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = tmp_path / "x.bin"
    f.write_bytes(b"tampered")
    with pytest.raises(ChecksumError):
        verify(f, "0" * 64)


def test_manifest_complete(repo_paths, params) -> None:  # type: ignore[no-untyped-def]
    m = read_json(repo_paths.manifest)
    assert m["sha256"] == params.data.expected_sha256
    for key in ("source_url", "licence", "retrieved_at", "row_count", "doi"):
        assert m.get(key) not in (None, "", "CALCULATED_VALUE")


@pytest.mark.data
def test_silver_quality(repo_paths, params) -> None:  # type: ignore[no-untyped-def]
    df = pd.read_parquet(repo_paths.silver_lines)
    assert check_silver(df, params.cleaning.service_stock_codes) == []


@pytest.mark.data
def test_gold_quality(repo_paths) -> None:  # type: ignore[no-untyped-def]
    assert check_gold(pd.read_parquet(repo_paths.gold / "retention.parquet")) == []


@pytest.mark.data
def test_bronze_row_count_matches_manifest(repo_paths) -> None:  # type: ignore[no-untyped-def]
    import pyarrow.parquet as pq

    m = read_json(repo_paths.manifest)
    assert pq.ParquetFile(repo_paths.bronze_transactions).metadata.num_rows == m["row_count"]
