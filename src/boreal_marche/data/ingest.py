"""Bronze layer: acquire the UCI file, verify SHA-256, convert to Parquet with an explicit schema.

Run with ``make data`` (``python -m boreal_marche.data.ingest``). The command is idempotent:
if the raw file already exists and its checksum matches, it is not re-downloaded.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.utils import sha256_file, utc_now, write_json

# Source column -> (bronze column, pandas dtype used when reading the Excel sheet).
SOURCE_COLUMNS: dict[str, tuple[str, str]] = {
    "Invoice": ("invoice", "string"),
    "StockCode": ("stock_code", "string"),
    "Description": ("description", "string"),
    "Quantity": ("quantity", "int32"),
    "InvoiceDate": ("invoice_date", "datetime64[ns]"),
    "Price": ("price", "float64"),
    "Customer ID": ("customer_id", "Int64"),
    "Country": ("country", "string"),
}

BRONZE_SCHEMA = pa.schema(
    [
        pa.field("row_id", pa.int64(), nullable=False),
        pa.field("invoice", pa.string(), nullable=False),
        pa.field("stock_code", pa.string(), nullable=False),
        pa.field("description", pa.string()),
        pa.field("quantity", pa.int32(), nullable=False),
        pa.field("invoice_date", pa.timestamp("ns"), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("customer_id", pa.int64()),
        pa.field("country", pa.string()),
        pa.field("source_sheet", pa.string(), nullable=False),
    ]
)


class ChecksumError(RuntimeError):
    """Raised when the raw file does not match the SHA-256 recorded in ``params.yaml``."""


def _download_xlsx(url: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "online_retail_ii.zip"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, zpath)  # noqa: S310 - fixed https URL from params.yaml
        with zipfile.ZipFile(zpath) as zf:
            member = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
            with zf.open(member) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)


def acquire(params: Params, paths: Paths) -> tuple[Path, str]:
    """Make sure the raw xlsx exists in ``data/raw``; return its path and how it was obtained."""
    raw = paths.root / params.data.raw_file
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        return raw, "existing data/raw copy"
    for cand in params.data.local_candidates:
        src = paths.root / cand
        if src.exists():
            shutil.copy2(src, raw)
            return raw, f"copied from local file {cand}"
    _download_xlsx(params.data.source_url, raw)
    return raw, f"downloaded from {params.data.source_url}"


def verify(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def read_raw(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read both yearly sheets with explicit dtypes and concatenate them verbatim."""
    read_dtypes = {c: "string" for c, (_, t) in SOURCE_COLUMNS.items() if t == "string"}
    sheets = pd.read_excel(path, sheet_name=None, dtype=read_dtypes, engine="openpyxl")
    frames = []
    counts: dict[str, int] = {}
    for name, df in sheets.items():
        missing = set(SOURCE_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"sheet {name!r} is missing columns {sorted(missing)}")
        df = df[list(SOURCE_COLUMNS)].rename(columns={k: v for k, (v, _) in SOURCE_COLUMNS.items()})
        df["source_sheet"] = name
        counts[name] = len(df)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "row_id", range(len(out)))
    for _, (col, dtype) in SOURCE_COLUMNS.items():
        out[col] = out[col].astype(dtype)  # type: ignore[call-overload]
    return out, counts


def to_bronze(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=BRONZE_SCHEMA, preserve_index=False)
    pq.write_table(table, dest)


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, object]:
    params = params or load_params()
    paths = paths or Paths()
    raw, how = acquire(params, paths)
    checksum = verify(raw, params.data.expected_sha256)
    print(f"SHA-256 verified: {checksum}")
    df, counts = read_raw(raw)
    to_bronze(df, paths.bronze_transactions)
    manifest: dict[str, object] = {
        "dataset": "UCI Online Retail II",
        "source": "UCI Machine Learning Repository",
        "source_url": params.data.source_url,
        "doi": params.data.doi,
        "licence": params.data.licence,
        "citation": "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5CG6D",
        "raw_file": params.data.raw_file,
        "raw_bytes": raw.stat().st_size,
        "sha256": checksum,
        "acquisition": how,
        "retrieved_at": utc_now(),
        "sheet_row_counts": counts,
        "row_count": int(len(df)),
        "date_min": str(df["invoice_date"].min()),
        "date_max": str(df["invoice_date"].max()),
        "bronze_file": str(paths.bronze_transactions.relative_to(paths.root)),
        "bronze_schema": {f.name: str(f.type) for f in BRONZE_SCHEMA},
        "scenario_note": "Public dataset re-skinned as the fictional retailer Boréal Marché for a "
        "pedagogical scenario. It is not proprietary client data.",
    }
    write_json(paths.manifest, manifest)
    print(f"Bronze rows: {len(df):,} -> {paths.bronze_transactions}")
    return manifest


if __name__ == "__main__":
    run()
