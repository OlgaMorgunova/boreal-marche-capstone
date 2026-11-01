"""Head A campaign policy: the fairness mitigation adopted in D5 §7, built and served (D7).

The automatic proposal is unchanged: ``p_return <= threshold``. Two things are added on top of it.

* **High-value review list.** Customers in the top two spend quartiles whom the single threshold
  does not propose but the equal-opportunity thresholds of M2 would (``threshold < p <= group
  threshold``). They get no coupon automatically: they are listed for the marketing reviewer, who
  approves every campaign list anyway (GOVERNANCE_LOI25.md §7). Spend is used to route a case to a
  person, never to change the automatic decision.
* **Randomised hold-out.** A fixed share of *proposed* customers receive no coupon this campaign,
  chosen by a hash of (campaign id, customer reference): no state to store, reproducible by an
  auditor, independent of every feature. The return rate of held-out versus treated customers
  measures the save rate that every dollar figure assumes and nobody has measured (D6 §12).

Spend edges and group thresholds come from ``fairness.json`` (fitted on train and validation only)
and are frozen into ``artifacts/campaign_policy.json``, which the API loads at startup; a policy
written for another model version is refused rather than applied.

``python -m boreal_marche.retention.campaign`` writes the artifact and scores the policy on the
validation and test snapshots, with the hold-out's power (``reports/metrics/campaign_policy.json``).
"""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from boreal_marche.config import CampaignParams, Paths, load_params
from boreal_marche.retention.fairness import SPEND_LABELS
from boreal_marche.utils import read_json, utc_now, write_json

Action = Literal["coupon", "holdout", "review", "none"]
ACTIONS: tuple[Action, ...] = ("coupon", "holdout", "review", "none")
Z_ALPHA, Z_POWER = 1.959964, 0.841621  # two-sided 5%, 80% power


@dataclass(frozen=True)
class CampaignPolicy:
    campaign_id: str
    model_version: str
    threshold: float
    spend_edges: list[float]
    review_thresholds: dict[str, float]
    holdout_share: float
    source: str

    @classmethod
    def from_fairness(cls, fairness: dict[str, Any], c: CampaignParams) -> CampaignPolicy:
        th = fairness["group_thresholds_M2"]
        return cls(
            campaign_id=c.campaign_id,
            model_version=fairness["model_version"],
            threshold=float(fairness["threshold"]),
            spend_edges=[float(x) for x in fairness["spend_edges_train"]],
            review_thresholds={q: float(th[q]) for q in c.review_quartiles},
            holdout_share=c.holdout_share,
            source=f"reports/metrics/fairness.json ({fairness['generated_at']})",
        )

    @classmethod
    def load(cls, path: Path) -> CampaignPolicy:
        return cls(**read_json(path))

    def spend_quartile(self, revenue_180d: float) -> str:
        """Same bins as the audit: ``pd.cut`` with right-closed edges."""
        return SPEND_LABELS[bisect.bisect_left(self.spend_edges, revenue_180d)]

    def in_holdout(self, customer_ref: str | None) -> bool:
        """Deterministic and feature-blind; a request without a reference is never held out."""
        if not customer_ref:
            return False
        h = hashlib.sha256(f"{self.campaign_id}:{customer_ref}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64 < self.holdout_share

    def action(self, p_return: float, revenue_180d: float, customer_ref: str | None) -> Action:
        if p_return <= self.threshold:
            return "holdout" if self.in_holdout(customer_ref) else "coupon"
        review_th = self.review_thresholds.get(self.spend_quartile(revenue_180d))
        return "review" if review_th is not None and p_return <= review_th else "none"


def write_policy(paths: Paths | None = None) -> CampaignPolicy:
    paths = paths or Paths()
    policy = CampaignPolicy.from_fairness(
        read_json(paths.metrics / "fairness.json"), load_params().campaign
    )
    write_json(paths.campaign_policy, asdict(policy))
    return policy


# ---------------------------------------------------------------------------- evaluation ---


def holdout_power(n_treated: int, n_holdout: int, return_rate: float, lapse_rate: float) -> float:
    """Smallest save rate detectable at 5% (two-sided) with 80% power. A coupon that saves a share
    ``s`` of lapsers raises the return rate by ``s * lapse_rate``; the test compares two return
    rates, so the detectable difference is divided by the lapse rate."""
    if not (n_treated and n_holdout and lapse_rate):
        return math.inf
    v = return_rate * (1 - return_rate) * (1 / n_treated + 1 / n_holdout)
    return (Z_ALPHA + Z_POWER) * math.sqrt(v) / lapse_rate


def score_snapshot(policy: CampaignPolicy, df: pd.DataFrame, p: np.ndarray) -> dict[str, Any]:
    from boreal_marche.monitoring.reference import customer_ref
    from boreal_marche.retention.profit import customer_economics, realised_profit

    b = load_params().business
    value, discount = customer_economics(df, b)
    y = df["y"].to_numpy()
    profit = realised_profit(y, value, discount, b)  # of contacting each customer, as assumed
    lapser = y == 0
    refs = [customer_ref(int(c)) for c in df["customer_id"]]
    act = np.array(
        [
            policy.action(float(pi), float(r), ref)
            for pi, r, ref in zip(p, df["revenue_180d"], refs, strict=True)
        ]
    )
    q = np.array([policy.spend_quartile(float(r)) for r in df["revenue_180d"]])
    proposed = np.isin(act, ["coupon", "holdout"])
    review = act == "review"

    def recall(m: np.ndarray, sel: np.ndarray) -> float | None:
        n = int((m & lapser).sum())
        return float((sel & m & lapser).sum() / n) if n else None

    by_quartile = {
        g: {
            "n": int((q == g).sum()),
            "lapser_recall_automatic": recall(q == g, proposed),
            "lapser_recall_if_review_approved": recall(q == g, proposed | review),
            "review_list": int((review & (q == g)).sum()),
        }
        for g in SPEND_LABELS
    }
    hold, treat = act == "holdout", act == "coupon"
    return_rate = float(y[proposed].mean()) if proposed.any() else 0.0
    lapse_prop = 1 - return_rate
    mde = holdout_power(int(treat.sum()), int(hold.sum()), return_rate, lapse_prop)
    # Quarters of the same size needed before a 20% save rate is detectable (variance ~ 1/k).
    target = b.save_rate
    quarters = math.ceil((mde / target) ** 2) if math.isfinite(mde) else None
    # The business rule as a coupon policy ("no order in 60 days", the D3 baseline), for Q9.
    rule = df["recency_days"].to_numpy() > 60
    return {
        "n": len(df),
        "actions": {a: int((act == a).sum()) for a in ACTIONS},
        "lapser_recall_model": recall(np.ones(len(df), bool), proposed),
        "rule_recency60": {
            "coupons": int(rule.sum()),
            "lapser_recall": recall(np.ones(len(df), bool), rule),
            "profit": float(profit[rule].sum()),
        },
        "review_list": {
            "n": int(review.sum()),
            "lapsers": int((review & lapser).sum()),
            "lapser_share": float(lapser[review].mean()) if review.any() else None,
            "profit_if_all_approved": float(profit[review].sum()),
        },
        "by_spend_quartile": by_quartile,
        "holdout": {
            "n": int(hold.sum()),
            "share_of_proposed": float(hold.sum() / proposed.sum()) if proposed.any() else None,
            # Randomisation check: the hash ignores features, so both arms should look alike.
            "mean_p_holdout": float(p[hold].mean()) if hold.any() else None,
            "mean_p_treated": float(p[treat].mean()) if treat.any() else None,
            "lapse_rate_holdout": float(lapser[hold].mean()) if hold.any() else None,
            "lapse_rate_treated": float(lapser[treat].mean()) if treat.any() else None,
            # The price of measuring: profit the withheld coupons would have made (as assumed).
            "profit_forgone": float(profit[hold].sum()),
            "profit_proposed": float(profit[proposed].sum()),
            "min_detectable_save_rate_one_quarter": mde,
            "quarters_to_detect_assumed_save_rate": quarters,
        },
    }


def run() -> dict[str, Any]:
    import joblib

    from boreal_marche.retention.artifact import RetentionModel
    from boreal_marche.retention.cv import load_gold

    paths = Paths()
    policy = write_policy(paths)
    model: RetentionModel = joblib.load(paths.model_file)
    if model.version != policy.model_version:
        raise RuntimeError(f"fairness.json is for {policy.model_version}, model is {model.version}")
    gold = load_gold(paths)
    out: dict[str, Any] = {"generated_at": utc_now(), "policy": asdict(policy), "snapshots": {}}
    for snap in ("validation", "test"):
        df = gold[gold["split"] == snap].reset_index(drop=True)
        p = model.predict_proba(df[model.feature_names])[:, 1]
        out["snapshots"][snap] = score_snapshot(policy, df, p)
    write_json(paths.metrics / "campaign_policy.json", out)
    return out


if __name__ == "__main__":
    r = run()
    for snap, s in r["snapshots"].items():
        h = s["holdout"]
        q4 = s["by_spend_quartile"]["Q4 (top)"]
        print(
            f"{snap}: {s['actions']} · review lapser share {s['review_list']['lapser_share']:.2f} "
            f"· Q4 recall {q4['lapser_recall_automatic']:.2f} -> "
            f"{q4['lapser_recall_if_review_approved']:.2f} · hold-out MDE "
            f"{h['min_detectable_save_rate_one_quarter']:.2f} "
            f"({h['quarters_to_detect_assumed_save_rate']} quarters for the assumed save rate)"
        )
