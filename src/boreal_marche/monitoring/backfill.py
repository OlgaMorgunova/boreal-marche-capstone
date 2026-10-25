"""Delayed ground truth: join logged predictions with the label once its 90-day window has closed.

Design (production): a weekly job selects predictions logged more than ``windows.label_days`` ago
that have no label yet, looks up each customer's orders in ``(prediction time, prediction time +
90 days]`` in the order database, and writes ``label = 1`` if there was at least one order. The
first true accuracy signal therefore arrives **90 days after the first logged prediction, plus up
to 7 days of job cadence**. Until then the leading indicators are input drift, prediction drift and
data quality (``/monitoring/report``).

Demonstration (this repository): the demo customers are real customers of the validation snapshot
whose 90-day outcome is already known historically, so ``--replay-validation`` writes those labels
for logged predictions of demo customers. The shifted cohort uses synthetic references and stays
unlabelled. The report marks replayed labels as such; they are in-sample for the calibrator and are
shown to prove the pipeline, not as a new performance estimate.
"""

from __future__ import annotations

import argparse
import os
import time

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.monitoring.drift import realised_metrics
from boreal_marche.monitoring.reference import customer_ref
from boreal_marche.monitoring.store import LogStore
from boreal_marche.retention.cv import load_gold


def replay_validation_labels(store: LogStore, params: Params, paths: Paths) -> int:
    gold = load_gold(paths)
    val = gold[gold["split"] == "validation"]
    labels = {
        customer_ref(int(c)): int(y) for c, y in zip(val["customer_id"], val["y"], strict=True)
    }
    preds = store.recent("prediction_log", 1_000_000)
    if preds.empty:
        return 0
    window = params.windows.label_days * 86400
    rows = [
        {
            "prediction_id": int(pid),
            "customer_ref": str(ref),
            "label": labels[str(ref)],
            "label_window_end": float(ts) + window,
            "observed_at": time.time(),
            "source": f"replay: validation snapshot {params.windows.cutoffs.validation[0]}",
        }
        for pid, ref, ts in zip(preds["id"], preds["customer_ref"], preds["ts"], strict=True)
        if str(ref) in labels
    ]
    return store.upsert_ground_truth(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--replay-validation", action="store_true")
    args = ap.parse_args()
    params, paths = load_params(), Paths()
    store = LogStore(args.database_url or os.environ.get("DATABASE_URL", "sqlite:///./boreal.db"))
    if args.replay_validation:
        print(f"labels written: {replay_validation_labels(store, params, paths)}")
    print(realised_metrics(store, params.windows.label_days))
    store.close()


if __name__ == "__main__":
    main()
