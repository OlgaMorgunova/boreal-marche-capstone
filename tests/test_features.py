"""Point-in-time correctness of the gold feature table (the leakage tests)."""

from __future__ import annotations

import pandas as pd
import pytest

from boreal_marche.features import FEATURE_SPEC, FEATURES, NON_FEATURE_COLS, build_snapshot
from tests.conftest import make_lines

CUTOFF = pd.Timestamp("2011-01-01")


def snap(lines: pd.DataFrame, canc: pd.DataFrame, offset: int = 0) -> pd.DataFrame:
    return build_snapshot(lines, canc, CUTOFF, 180, 90, offset).set_index("customer_id")


def test_at_least_25_documented_features() -> None:
    assert len(FEATURES) >= 25
    assert len(set(FEATURES)) == len(FEATURES)
    assert all(f.definition and f.rationale and f.missing for f in FEATURE_SPEC)
    assert not set(NON_FEATURE_COLS) & set(FEATURES)


def test_label_is_future_purchase_within_window(synthetic_lines, empty_cancellations) -> None:
    s = snap(synthetic_lines, empty_cancellations)
    assert s.loc[1, "y"] == 1  # bought on 2011-02-01
    assert s.loc[2, "y"] == 0  # next purchase is after the 90-day window


def test_guests_excluded(synthetic_lines, empty_cancellations) -> None:
    s = snap(synthetic_lines, empty_cancellations)
    assert set(s.index) == {1, 2}


def test_features_ignore_rows_after_cutoff(synthetic_lines, empty_cancellations) -> None:
    """Adding (or removing) future rows must not change any feature value."""
    base = snap(synthetic_lines, empty_cancellations)[FEATURES]
    future = make_lines(
        [
            ("9999", 1, "Z", "2011-01-02 10:00", 1000, 99.0),
            ("9998", 2, "Z", "2011-03-01 10:00", 5, 1.0),
        ]
    )
    more = snap(pd.concat([synthetic_lines, future], ignore_index=True), empty_cancellations)[
        FEATURES
    ]
    pd.testing.assert_frame_equal(base, more)


def test_recency_and_counts(synthetic_lines, empty_cancellations) -> None:
    s = snap(synthetic_lines, empty_cancellations)
    last = pd.Timestamp("2010-11-20 12:00")
    assert s.loc[1, "recency_days"] == pytest.approx((CUTOFF - last).total_seconds() / 86400)
    assert s.loc[1, "orders_180d"] == 2
    assert s.loc[1, "revenue_180d"] == pytest.approx(2 * 5 + 3 + 4 * 5)
    assert s.loc[1, "mean_gap_days"] == pytest.approx(71.0)


def test_missing_cadence_is_nan_not_zero(synthetic_lines, empty_cancellations) -> None:
    s = snap(synthetic_lines, empty_cancellations)
    assert pd.isna(s.loc[2, "mean_gap_days"])  # single order day
    assert pd.isna(s.loc[2, "overdue_ratio"])


def test_leaky_variant_changes_features(synthetic_lines, empty_cancellations) -> None:
    """The deliberately leaky snapshot sees the label-window purchase — the test that would
    detect the leak if it were ever used for training."""
    clean = snap(synthetic_lines, empty_cancellations)
    leaky = snap(synthetic_lines, empty_cancellations, offset=90)
    assert leaky.loc[1, "orders_lifetime"] > clean.loc[1, "orders_lifetime"]


def test_cancellations_after_cutoff_ignored(synthetic_lines) -> None:
    canc = pd.DataFrame(
        {
            "customer_id": pd.array([1, 1], dtype="Int64"),
            "invoice_date": pd.to_datetime(["2010-12-01", "2011-01-15"]),
            "stock_code": ["A", "A"],
        }
    )
    s = snap(synthetic_lines, canc)
    assert s.loc[1, "cancel_lines_180d"] == 1
