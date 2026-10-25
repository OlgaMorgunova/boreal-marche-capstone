"""Traffic simulation and load test for the Boréal Marché API (Locust).

Adapted from the course example ``doc/curs_doc/Locust-Test/locustfile.py`` (HttpUser, weighted
@task, on_start health check, catch_response, grouped request names), extended to drive real model
traffic so the monitoring dashboard has something to watch.

Cohorts (``BOREAL_COHORT`` environment variable, default ``normal``):

* ``normal``  — real customers: feature vectors of the held-out validation-snapshot demo customers
  (``artifacts/demo_customers.json``); baskets built around real rule antecedents plus popular items.
* ``shifted`` — a deliberately shifted cohort that simulates a new acquisition campaign in the
  Maritimes: recent sign-ups (short tenure, one or two orders, no cadence history yet), lower spend,
  a different region mix, and baskets of new SKUs the rules have never seen.

Each request carries ``X-Traffic-Cohort`` so the demo can show which traffic was shifted; the drift
checks never read it.

Interactive (web UI on http://127.0.0.1:8089, like the course example):
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
Headless, as used by ``scripts/simulate_traffic.py``:
    BOREAL_COHORT=shifted locust -f loadtest/locustfile.py --headless -u 10 -r 5 -t 2m \
        --host http://127.0.0.1:8000 --csv reports/loadtest/shifted --html reports/loadtest/shifted.html
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
COHORT = os.environ.get("BOREAL_COHORT", "normal")
CUSTOMERS: list[dict[str, Any]] = json.loads(
    (ROOT / "artifacts" / "demo_customers.json").read_text(encoding="utf-8")
)["customers"]
_RULES = json.loads((ROOT / "artifacts" / "rules.json").read_text(encoding="utf-8"))
ANTECEDENTS: list[list[str]] = [r["antecedent"] for r in _RULES["rules"][:2000]]
POPULAR: list[str] = _RULES["popularity"][:100]
GOLDEN = json.loads(
    (ROOT / "tests" / "golden" / "assistant_golden.json").read_text(encoding="utf-8")
)["cases"]


def shifted_features(rng: random.Random) -> dict[str, Any]:
    """A new-customer profile: drawn from a real customer, then pushed off-distribution."""
    f = dict(rng.choice(CUSTOMERS)["features"])
    orders = rng.choice([1, 1, 2])
    tenure = rng.uniform(5, 60)
    revenue = f["revenue_180d"] * rng.uniform(0.1, 0.3)
    f.update(
        recency_days=rng.uniform(0, 30),
        tenure_days=tenure,
        tenure_censored=0,
        orders_30d=min(orders, 1),
        orders_90d=orders,
        orders_180d=orders,
        orders_lifetime=orders,
        active_months_180d=1,
        revenue_30d=revenue,
        revenue_90d=revenue,
        revenue_180d=revenue,
        revenue_lifetime=revenue,
        aov_180d=revenue / orders,
        max_order_value_180d=revenue / orders,
        mean_gap_days=None if orders == 1 else rng.uniform(3, 20),
        std_gap_days=None,
        overdue_ratio=None if orders == 1 else rng.uniform(0.5, 3),
        orders_trend=0.0,
        revenue_trend=None,
        region=rng.choice(["MARITIMES", "MARITIMES", "MARITIMES", "ON"]),
    )
    return f


class Shopper(HttpUser):
    """One simulated client of the platform (storefront + assistant)."""

    wait_time = between(0.2, 1.0)

    def on_start(self) -> None:
        self.rng = random.Random()
        self.headers = {"X-Traffic-Cohort": COHORT}
        self.client.get("/health", headers=self.headers)

    def _features(self) -> tuple[str, dict[str, Any]]:
        if COHORT == "shifted":
            return f"NEW-{self.rng.randrange(10**6):06d}", shifted_features(self.rng)
        c = self.rng.choice(CUSTOMERS)
        return c["ref"], c["features"]

    def _basket(self) -> list[str]:
        if COHORT == "shifted":
            new = [f"N{self.rng.randrange(10**4):04d}" for _ in range(self.rng.randint(1, 3))]
            return new + ([self.rng.choice(POPULAR)] if self.rng.random() < 0.1 else [])
        return list(self.rng.choice(ANTECEDENTS)) + self.rng.sample(POPULAR, self.rng.randint(0, 2))

    @task(6)
    def predict(self) -> None:
        ref, feats = self._features()
        with self.client.post(
            "/predict",
            json={"customer_ref": ref, "features": feats},
            headers=self.headers,
            name="/predict",
            catch_response=True,
        ) as r:
            if r.status_code == 200 and 0 <= r.json()["p_return"] <= 1:
                r.success()
            else:
                r.failure(f"status={r.status_code} body={r.text[:200]}")

    @task(3)
    def recommend(self) -> None:
        with self.client.post(
            "/recommend",
            json={"basket": self._basket(), "k": 5},
            headers=self.headers,
            name="/recommend",
            catch_response=True,
        ) as r:
            if r.status_code == 200 and r.json()["recommendations"]:
                r.success()
            else:
                r.failure(f"status={r.status_code}")

    @task(1)
    def predict_batch(self) -> None:
        items = [
            {"customer_ref": ref, "features": f}
            for ref, f in (self._features() for _ in range(self.rng.randint(5, 20)))
        ]
        self.client.post(
            "/predict/batch", json={"items": items}, headers=self.headers, name="/predict/batch"
        )

    @task(1)
    def chat(self) -> None:
        case = self.rng.choice(GOLDEN)
        self.client.post(
            "/chat",
            json={
                "message": case["question"],
                "provider": "rule",
                "customer_ref": CUSTOMERS[0]["ref"],
            },
            headers=self.headers,
            name="/chat",
        )

    @task(1)
    def version(self) -> None:
        self.client.get("/version", headers=self.headers, name="/version")
