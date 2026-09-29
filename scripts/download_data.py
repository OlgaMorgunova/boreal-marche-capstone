from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)

OUTPUT = DATA_DIR / "online_retail_II.zip"


def main() -> None:
    print(f"Downloading dataset from:\n{URL}")

    urlretrieve(URL, OUTPUT)

    print(f"Saved to: {OUTPUT}")


if __name__ == "__main__":
    main()