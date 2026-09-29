"""Offline evaluation of Head B: leave-one-out hit-rate@k and coverage vs global popularity.

Protocol: for every evaluation basket with at least two distinct items, hide one item chosen
uniformly at random (fixed seed), recommend ``k`` items from the rest, and count a hit if the hidden
item is recommended. The same hidden item is used for the rules engine and for the baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from boreal_marche.config import Params, Paths, load_params
from boreal_marche.recommend.baskets import basket_lists, invoice_items, load_lines
from boreal_marche.recommend.engine import popularity_recommend
from boreal_marche.utils import utc_now, write_json

if TYPE_CHECKING:
    from boreal_marche.recommend.engine import RuleEngine


def leave_one_out(pairs: pd.DataFrame, seed: int) -> list[tuple[set[str], str]]:
    rng = np.random.default_rng(seed)
    cases = []
    for items in basket_lists(pairs, min_items=2):
        hidden = items[int(rng.integers(len(items)))]
        cases.append(({i for i in items if i != hidden}, hidden))
    return cases


def hit_rate(engine: RuleEngine, pairs: pd.DataFrame, k: int, seed: int) -> dict[str, Any]:
    cases = leave_one_out(pairs, seed)
    hits_rules = hits_hybrid = hits_pop = fired = 0
    recommended: set[str] = set()
    for basket, hidden in cases:
        res = engine.recommend(sorted(basket), k=k, fill=True)
        items = [r.item for r in res.recommendations]
        rule_items = [r.item for r in res.recommendations if r.source == "rule"]
        recommended.update(items)
        fired += res.strategy in ("rules", "rules+popularity")
        hits_rules += hidden in rule_items
        hits_hybrid += hidden in items
        hits_pop += hidden in popularity_recommend(engine.popularity, basket, k)
    n = max(1, len(cases))
    return {
        "n_cases": len(cases),
        "hit_rate_rules_only": hits_rules / n,
        "hit_rate_hybrid": hits_hybrid / n,
        "hit_rate_popularity": hits_pop / n,
        "basket_coverage": fired / n,
        "catalogue_coverage": len(recommended) / max(1, len(engine.catalogue)),
        "distinct_items_recommended": len(recommended),
    }


def strategy_examples(engine: RuleEngine) -> dict[str, Any]:
    """Documented behaviour for the empty basket, a no-match basket and a matching basket."""
    first = engine.rules[0] if engine.rules else None
    cases = {
        "empty_basket": [],
        "no_match_basket": ["NOT-A-REAL-SKU"],
        "matching_basket": sorted(first.antecedent) if first else [],
    }
    out: dict[str, Any] = {}
    for name, basket in cases.items():
        res = engine.recommend(basket, k=5)
        out[name] = {
            "basket": basket,
            "strategy": res.strategy,
            "unknown_items": res.unknown_items,
            "recommendations": [
                {
                    "item": r.item,
                    "description": r.description,
                    "source": r.source,
                    "because": sorted(r.rule.antecedent) if r.rule else None,
                    "lift": r.rule.lift if r.rule else None,
                }
                for r in res.recommendations
            ],
        }
    return out


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    from boreal_marche.recommend.engine import RuleEngine

    params = params or load_params()
    paths = paths or Paths()
    rp = params.recommend
    engine = RuleEngine.from_json(paths.rules_file, min_lift=rp.min_lift)
    pairs = invoice_items(load_lines(paths))
    hold = pairs[pairs["invoice_date"] >= pd.Timestamp(rp.holdout_start)]
    ev = hit_rate(engine, hold, rp.top_k, params.seed)
    lift_ratio = (
        ev["hit_rate_hybrid"] / ev["hit_rate_popularity"] if ev["hit_rate_popularity"] else None
    )
    payload = {
        "generated_at": utc_now(),
        "holdout_start": rp.holdout_start,
        "k": rp.top_k,
        **ev,
        "hybrid_vs_popularity_ratio": lift_ratio,
        "strategies": strategy_examples(engine),
        "rule_count": len(engine.rules),
    }
    write_json(paths.metrics / "rules_evaluation.json", payload)
    print({k: v for k, v in payload.items() if k != "strategies"})
    return payload


if __name__ == "__main__":
    run()
