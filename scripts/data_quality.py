from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/online_retail_II.parquet"
)


def main() -> None:
    df = pd.read_parquet(INPUT_FILE)

    print("=== DATASET OVERVIEW ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n=== MISSING VALUES ===")
    missing = df.isna().sum()
    print(missing[missing > 0])

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== DUPLICATES ===")
    print(
        f"Exact duplicate rows: "
        f"{df.duplicated().sum():,}"
    )

    print("\n=== CANCELLATIONS ===")

    cancellations = df["InvoiceNo"].astype(str).str.startswith("C")

    print(
        f"Cancellation rows: "
        f"{cancellations.sum():,}"
    )

    print("\n=== QUANTITY ===")
    print(df["Quantity"].describe())

    print("\n=== UNIT PRICE ===")
    print(df["UnitPrice"].describe())


if __name__ == "__main__":
    main()