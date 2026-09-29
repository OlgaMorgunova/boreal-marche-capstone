from hashlib import sha256
from pathlib import Path


FILE = Path("data/raw/online_retail_II.xlsx")


def calculate_sha256(path: Path) -> str:
    digest = sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    checksum = calculate_sha256(FILE)

    print("SHA-256:")
    print(checksum)


if __name__ == "__main__":
    main()