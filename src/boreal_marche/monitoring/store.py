"""Prediction and request log (SQLAlchemy Core; SQLite locally and on App Runner, Postgres in
Compose).

Every API request, prediction, recommendation and chat turn is written to a table: the monitoring
dashboard is built from these rows. Writes go through a queue drained by one background thread in
small batches, so logging never sits on the request path and SQLite sees a single writer.

``cohort`` is a label the *traffic simulator* sends (``X-Traffic-Cohort``) so the demonstration can
show which traffic was deliberately shifted. Drift checks never read it — real traffic has no label.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

request_log = Table(
    "request_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, index=True, nullable=False),
    Column("request_id", String(36)),
    Column("method", String(8)),
    Column("path", String(64), index=True),
    Column("status", Integer),
    Column("latency_ms", Float),
    Column("caller", String(16)),
    Column("cohort", String(16)),
)
prediction_log = Table(
    "prediction_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, index=True, nullable=False),
    Column("request_id", String(36)),
    Column("customer_ref", String(64)),
    Column("model_version", String(64)),
    Column("p_return", Float),
    Column("send_coupon", Boolean),
    Column("threshold", Float),
    Column("features", Text),
    Column("cohort", String(16)),
)
recommendation_log = Table(
    "recommendation_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, index=True, nullable=False),
    Column("request_id", String(36)),
    Column("basket_size", Integer),
    Column("n_unknown", Integer),
    Column("strategy", String(32)),
    Column("n_items", Integer),
    Column("cohort", String(16)),
)
chat_log = Table(
    "chat_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, index=True, nullable=False),
    Column("request_id", String(36)),
    Column("provider", String(16)),
    Column("tool", String(32)),
    Column("lang", String(4)),
    Column("refused", Boolean),
    Column("handoff", Boolean),
    Column("latency_ms", Float),
    Column("cohort", String(16)),
)
alert_log = Table(
    "alert_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, index=True, nullable=False),
    Column("check_name", String(64)),
    Column("level", String(24)),
    Column("previous", String(24)),
    Column("value", Float, nullable=True),
    Column("message", Text),
    Column("notified", String(64)),
)
# Delayed ground truth: filled by the backfill job once the 90-day label window has closed.
ground_truth = Table(
    "ground_truth",
    metadata,
    Column("prediction_id", Integer, primary_key=True),
    Column("customer_ref", String(64)),
    Column("label", Integer),
    Column("label_window_end", Float),
    Column("observed_at", Float),
    Column("source", String(64)),
)
TABLES = {
    t.name: t
    for t in (request_log, prediction_log, recommendation_log, chat_log, alert_log, ground_truth)
}


class LogStore:
    def __init__(self, url: str) -> None:
        self.url = url
        sqlite = url.startswith("sqlite")
        self.engine: Engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if sqlite else {},
            pool_pre_ping=not sqlite,
        )
        if sqlite:

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn: Any, _: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.close()

        metadata.create_all(self.engine)
        self._q: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="log-writer", daemon=True)
        self._thread.start()

    @property
    def kind(self) -> str:
        return self.engine.dialect.name

    # ----------------------------------------------------------------------------- writes ---

    def log(self, table: str, row: dict[str, Any]) -> None:
        row.setdefault("ts", time.time())
        self._q.put((table, row))

    def _writer(self) -> None:
        stop = False
        while not stop:
            items = [self._q.get()]
            while len(items) < 500:
                try:
                    items.append(self._q.get_nowait())
                except queue.Empty:
                    break
            batch = [i for i in items if i is not None]
            stop = len(batch) < len(items)
            try:
                with self.engine.begin() as conn:
                    for name in {t for t, _ in batch}:
                        conn.execute(insert(TABLES[name]), [r for t, r in batch if t == name])
            except Exception as exc:  # noqa: BLE001 - logging must never take the API down
                print(json.dumps({"event": "log_write_failed", "error": str(exc)[:200]}))
            finally:
                for _ in items:
                    self._q.task_done()

    def flush(self) -> None:
        self._q.join()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=5)
        self.engine.dispose()

    def insert_alert(self, row: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(alert_log), [row | {"ts": row.get("ts", time.time())}])

    def upsert_ground_truth(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(
                delete(ground_truth).where(
                    ground_truth.c.prediction_id.in_([r["prediction_id"] for r in rows])
                )
            )
            conn.execute(insert(ground_truth), rows)
        return len(rows)

    def purge(self, older_than_days: float) -> dict[str, int]:
        """Retention policy: prediction logs are kept 13 months (privacy notice)."""
        cutoff = time.time() - older_than_days * 86400
        out = {}
        with self.engine.begin() as conn:
            for t in (request_log, prediction_log, recommendation_log, chat_log):
                out[t.name] = conn.execute(delete(t).where(t.c.ts < cutoff)).rowcount
        return out

    # ------------------------------------------------------------------------------ reads ---

    def recent(self, table: str, n: int) -> pd.DataFrame:
        t = TABLES[table]
        q = select(t).order_by(t.c.id.desc()).limit(n)
        with self.engine.connect() as conn:
            df = pd.DataFrame(conn.execute(q).mappings().all())
        return df.iloc[::-1].reset_index(drop=True) if len(df) else df

    def count(self, table: str) -> int:
        t = TABLES[table]
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(t)).scalar_one())

    def last_levels(self) -> dict[str, str]:
        """Most recent alert level per check (the alert state machine's memory)."""
        df = self.recent("alert_log", 1000)
        if df.empty:
            return {}
        return {str(k): str(v) for k, v in df.groupby("check_name")["level"].last().items()}

    def labelled_predictions(self) -> pd.DataFrame:
        q = select(prediction_log.c.id, prediction_log.c.ts, prediction_log.c.p_return,
                   prediction_log.c.send_coupon, ground_truth.c.label).join(
            ground_truth, ground_truth.c.prediction_id == prediction_log.c.id
        )  # fmt: skip
        with self.engine.connect() as conn:
            return pd.DataFrame(conn.execute(q).mappings().all())
