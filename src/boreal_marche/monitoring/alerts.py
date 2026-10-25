"""Alerting loop: a check that changes level writes an alert row and notifies.

The API evaluates the report every ``monitoring.check_interval_seconds`` in a background thread.
An alert is raised on a *transition* (green -> amber, amber -> red, red -> green ...), not on every
evaluation, so a sustained breach notifies once. Notification channels:

* always: a structured ``alert`` log line (stdout -> CloudWatch Logs on App Runner) and an
  ``alert_log`` row shown on the dashboard;
* ``ALERT_WEBHOOK_URL`` set: a JSON POST (Slack / Microsoft Teams / Discord incoming webhook);
* the scheduled GitHub workflow ``monitor.yml`` polls ``/monitoring/report`` and opens an issue when
  the deployed service is red.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from boreal_marche.monitoring.store import LogStore


def notify(row: dict[str, Any], webhook: str | None) -> str:
    text = (
        f"[Boréal Marché monitoring] {row['check_name']}: {row['previous']} -> {row['level']}"
        f" (value={row['value']}) {row['message']}"
    )
    print(json.dumps({"event": "alert", **row}, default=str), flush=True)
    if not webhook:
        return "log"
    try:
        httpx.post(webhook, json={"text": text, "content": text}, timeout=5).raise_for_status()
        return "log+webhook"
    except httpx.HTTPError as exc:
        return f"log (webhook failed: {type(exc).__name__})"


def evaluate(store: LogStore, report: dict[str, Any]) -> list[dict[str, Any]]:
    """Record and notify every check whose level changed since its last alert."""
    previous = store.last_levels()
    webhook = os.environ.get("ALERT_WEBHOOK_URL") or None
    raised = []
    for check in report["checks"]:
        level = check["status"]
        before = previous.get(check["name"], "green")
        if level == "insufficient_data" or level == before:
            continue
        row = {
            "ts": time.time(),
            "check_name": check["name"],
            "level": level,
            "previous": before,
            "value": check["value"],
            "message": f"{check['unit']}; thresholds {check['threshold']}"
            + (f"; {check['detail']}" if check.get("detail") else ""),
        }
        row["notified"] = notify(row, webhook)
        store.insert_alert(row)
        raised.append(row)
    return raised
