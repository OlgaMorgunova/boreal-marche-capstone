"""Head B serving engine behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from boreal_marche.recommend.baskets import build_matrix
from boreal_marche.recommend.engine import Rule, RuleEngine, popularity_recommend
from boreal_marche.recommend.evaluate import hit_rate

CAT = {k: f"ITEM {k}" for k in "ABCDEFG"}
RULES = [
    Rule(0, frozenset({"A"}), "B", 0.1, 0.5, 2.0),
    Rule(1, frozenset({"A", "C"}), "B", 0.05, 0.6, 4.0),  # same consequent, higher lift
    Rule(2, frozenset({"A"}), "D", 0.1, 0.4, 3.0),
    Rule(3, frozenset({"E"}), "A", 0.1, 0.9, 1.1),  # below the lift floor
    Rule(4, frozenset({"C", "F"}), "G", 0.02, 0.7, 9.0),  # needs F too
]


def engine(**kw) -> RuleEngine:  # type: ignore[no-untyped-def]
    return RuleEngine(RULES, ["A", "B", "C", "D", "E"], CAT, min_lift=1.5, **kw)


def test_empty_basket_uses_popularity() -> None:
    res = engine().recommend([], k=3)
    assert res.strategy == "empty_basket"
    assert [r.item for r in res.recommendations] == ["A", "B", "C"]


def test_no_match_falls_back_and_reports_unknown() -> None:
    res = engine().recommend(["ZZZ"], k=2)
    assert res.strategy == "popularity_fallback"
    assert res.unknown_items == ["ZZZ"]


def test_dedupe_by_consequent_keeps_highest_lift_and_returns_rule() -> None:
    res = engine().recommend(["A", "C"], k=5, fill=False)
    b = next(r for r in res.recommendations if r.item == "B")
    assert b.rule is not None and b.rule.id == 1 and b.source == "rule"
    assert [r.item for r in res.recommendations].count("B") == 1


def test_excludes_basket_items_and_partial_antecedents() -> None:
    res = engine().recommend(["A", "B", "C"], k=5, fill=False)
    items = [r.item for r in res.recommendations]
    assert "B" not in items and "A" not in items
    assert "G" not in items  # rule 4 needs F


def test_lift_floor_applied() -> None:
    res = engine().recommend(["E"], k=5, fill=False)
    assert res.strategy == "popularity_fallback"


def test_fill_marks_strategy() -> None:
    res = engine().recommend(["A"], k=4)
    assert res.strategy == "rules+popularity"
    assert {r.source for r in res.recommendations} == {"rule", "popularity"}


def test_rank_by_confidence() -> None:
    res = engine(rank_by="confidence").recommend(["A"], k=2, fill=False)
    assert [r.item for r in res.recommendations] == ["B", "D"]


def test_popularity_baseline() -> None:
    assert popularity_recommend(["A", "B", "C"], {"A"}, 2) == ["B", "C"]


def test_from_json_roundtrip(tmp_path: Path) -> None:
    payload = {
        "header": {"mined_at": "x", "params": {"rank_by": "lift"}},
        "rules": [
            {
                "id": r.id,
                "antecedent": sorted(r.antecedent),
                "consequent": r.consequent,
                "support": r.support,
                "confidence": r.confidence,
                "lift": r.lift,
            }
            for r in RULES
        ],
        "popularity": ["A", "B"],
        "catalogue": CAT,
    }
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(payload))
    e = RuleEngine.from_json(p, min_lift=1.5)
    assert len(e.rules) == 4 and e.header["mined_at"] == "x"


def test_sparse_matrix_and_hit_rate() -> None:
    pairs = pd.DataFrame(
        {
            "invoice": ["1", "1", "2", "2", "3"],
            "stock_code": ["A", "B", "A", "D", "C"],
            "invoice_date": pd.to_datetime(["2011-01-01"] * 5),
        }
    )
    b = build_matrix(pairs)
    assert b.matrix.shape == (3, 4) and b.matrix.nnz == 5
    assert b.memory()["ratio"] > 0
    ev = hit_rate(engine(), pairs, k=5, seed=0)
    assert ev["n_cases"] == 2 and 0 <= ev["hit_rate_hybrid"] <= 1
