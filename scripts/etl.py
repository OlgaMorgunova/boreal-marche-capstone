from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

INPUT_FILE = RAW_DIR / "online_retail_II.xlsx"
OUTPUT_FILE = PROCESSED_DIR / "online_retail_II.parquet"


def load_data() -> pd.DataFrame:
    sheets = pd.read_excel(
        INPUT_FILE,
        sheet_name=None,
    )

    frames = []

    for sheet_name, df in sheets.items():
        print(f"Reading sheet: {sheet_name}")
        print(f"Rows: {len(df):,}")

        df["source_sheet"] = sheet_name
        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def main() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    print("\nFinal dataset:")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()