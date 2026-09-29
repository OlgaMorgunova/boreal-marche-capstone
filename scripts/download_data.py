"""Thin wrapper kept for backwards compatibility: acquire + verify + bronze. Prefer `make data`."""

from boreal_marche.data.ingest import run

if __name__ == "__main__":
    run()
