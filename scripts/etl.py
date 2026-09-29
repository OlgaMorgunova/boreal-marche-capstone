"""Thin wrapper: bronze -> silver -> gold. Prefer `make etl features`."""

from boreal_marche import features
from boreal_marche.data import cleaning, ingest

if __name__ == "__main__":
    ingest.run()
    cleaning.run()
    features.run()
