"""Head B baskets: clean invoices -> sparse invoice x item matrix.

Guest invoices (no customer id) are kept: for basket mining the invoice, not the customer, is the
unit of analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from boreal_marche.config import Paths


@dataclass
class Baskets:
    matrix: sparse.csr_matrix  # bool, invoices x items
    items: list[str]
    invoices: list[str]

    def memory(self) -> dict[str, float]:
        m = self.matrix
        sparse_bytes = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
        dense_bytes = m.shape[0] * m.shape[1]  # 1 byte per bool cell
        return {
            "n_baskets": int(m.shape[0]),
            "n_items": int(m.shape[1]),
            "nnz": int(m.nnz),
            "density": float(m.nnz / (m.shape[0] * m.shape[1])),
            "dense_bool_mb": dense_bytes / 2**20,
            "sparse_csr_mb": sparse_bytes / 2**20,
            "ratio": dense_bytes / sparse_bytes,
        }


def load_lines(paths: Paths) -> pd.DataFrame:
    cols = ["invoice", "stock_code", "description", "invoice_date"]
    return pd.read_parquet(paths.silver_lines, columns=cols)


def invoice_items(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per (invoice, item) with the invoice timestamp."""
    return (
        lines.groupby(["invoice", "stock_code"], sort=False, observed=True)["invoice_date"]
        .min()
        .reset_index()
    )


def build_matrix(pairs: pd.DataFrame, items: list[str] | None = None) -> Baskets:
    """Sparse matrix; ``items`` fixes the vocabulary (unknown items are dropped)."""
    if items is None:
        items = sorted(pairs["stock_code"].unique())
    col = pd.Index(items)
    pairs = pairs[pairs["stock_code"].isin(col)]
    inv = pd.Index(pairs["invoice"].unique())
    rows = inv.get_indexer(pd.Index(pairs["invoice"]))
    cols = col.get_indexer(pd.Index(pairs["stock_code"]))
    m = sparse.csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=(len(inv), len(col))
    )
    return Baskets(m, list(col), list(inv))


def basket_lists(pairs: pd.DataFrame, min_items: int = 1) -> pd.Series:
    """invoice -> sorted list of distinct items (baskets with at least ``min_items``)."""
    s = pairs.groupby("invoice", sort=False)["stock_code"].agg(lambda x: sorted(set(x)))
    return s[s.map(len) >= min_items]
