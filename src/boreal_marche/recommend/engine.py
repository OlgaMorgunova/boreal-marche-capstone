"""Online half of Head B: an antecedent-indexed rule engine over the static ``rules.json``.

Mining is offline and expensive; serving is a dictionary lookup plus subset checks. The engine:

* indexes rules by each antecedent item (only rules that could fire for the basket are checked);
* keeps a rule only if its whole antecedent is in the basket and its consequent is not;
* deduplicates by consequent, keeping the rule with the highest lift;
* returns the firing rule with each recommendation so the UI can say "because you added X";
* falls back to global popularity for an empty basket or a basket that matches no rule, and says
  which strategy fired.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Strategy = Literal["rules", "rules+popularity", "popularity_fallback", "empty_basket"]


@dataclass(frozen=True)
class Rule:
    id: int
    antecedent: frozenset[str]
    consequent: str
    support: float
    confidence: float
    lift: float


@dataclass
class Recommendation:
    item: str
    description: str
    score: float
    source: Literal["rule", "popularity"]
    rule: Rule | None = None


@dataclass
class Result:
    strategy: Strategy
    recommendations: list[Recommendation] = field(default_factory=list)
    unknown_items: list[str] = field(default_factory=list)
    rules_checked: int = 0


class RuleEngine:
    def __init__(
        self,
        rules: list[Rule],
        popularity: list[str],
        catalogue: dict[str, str],
        min_lift: float = 1.0,
        rank_by: str = "lift",
        header: dict[str, Any] | None = None,
    ) -> None:
        self.rules = [r for r in rules if r.lift >= min_lift]
        self.popularity = popularity
        self.catalogue = catalogue
        self.rank_by = rank_by
        self.header = header or {}
        self.by_item: dict[str, list[Rule]] = defaultdict(list)
        for r in self.rules:
            for a in r.antecedent:
                self.by_item[a].append(r)

    @classmethod
    def from_json(cls, path: Path, **kwargs: Any) -> RuleEngine:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rules = [
            Rule(
                r["id"],
                frozenset(r["antecedent"]),
                r["consequent"],
                r["support"],
                r["confidence"],
                r["lift"],
            )
            for r in payload["rules"]
        ]
        h = payload["header"]
        kwargs.setdefault("rank_by", h.get("params", {}).get("rank_by", "lift"))
        return cls(rules, payload["popularity"], payload["catalogue"], header=h, **kwargs)

    def _key(self, r: Rule) -> tuple[float, float]:
        return (r.lift, r.confidence) if self.rank_by == "lift" else (r.confidence, r.lift)

    def recommend(self, basket: list[str], k: int = 5, fill: bool = True) -> Result:
        items = {str(x).strip().upper() for x in basket if str(x).strip()}
        unknown = sorted(i for i in items if i not in self.catalogue)
        if not items:
            return Result("empty_basket", self._popular(items, k), unknown)
        best: dict[str, Rule] = {}
        checked = 0
        seen: set[int] = set()
        for it in items:
            for r in self.by_item.get(it, ()):
                if r.id in seen:
                    continue
                seen.add(r.id)
                checked += 1
                if r.consequent in items or not r.antecedent <= items:
                    continue
                cur = best.get(r.consequent)
                if cur is None or (r.lift, r.confidence) > (cur.lift, cur.confidence):
                    best[r.consequent] = r
        ranked = sorted(best.values(), key=self._key, reverse=True)[:k]
        recs = [
            Recommendation(r.consequent, self.catalogue.get(r.consequent, ""), r.lift, "rule", r)
            for r in ranked
        ]
        if not recs:
            return Result("popularity_fallback", self._popular(items, k), unknown, checked)
        strategy: Strategy = "rules"
        if fill and len(recs) < k:
            taken = items | {r.item for r in recs}
            recs += self._popular(taken, k - len(recs))
            strategy = "rules+popularity"
        return Result(strategy, recs, unknown, checked)

    def _popular(self, exclude: set[str], k: int) -> list[Recommendation]:
        out = []
        for rank, it in enumerate(self.popularity):
            if it not in exclude:
                out.append(
                    Recommendation(it, self.catalogue.get(it, ""), 1.0 / (rank + 1), "popularity")
                )
                if len(out) == k:
                    break
        return out


def popularity_recommend(popularity: list[str], basket: set[str], k: int) -> list[str]:
    """The global-popularity baseline: top-k bestsellers not already in the basket."""
    return [it for it in popularity if it not in basket][:k]
