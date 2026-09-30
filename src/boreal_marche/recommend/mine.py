"""Offline half of Head B: mine association rules and export the static ``rules.json``.

``python -m boreal_marche.recommend.mine sweep``  support x confidence x lift sweep, scored by
    hit-rate@5 on a validation slice *inside* the mining period (the holdout is not used).
``python -m boreal_marche.recommend.mine``        production mining with the chosen parameters
    (FP-Growth), Apriori timed on the same input for comparison, rules exported to JSON.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from boreal_marche import tracking
from boreal_marche.config import Params, Paths, load_params
from boreal_marche.recommend.baskets import (
    Baskets,
    build_matrix,
    invoice_items,
    load_lines,
)
from boreal_marche.recommend.engine import Rule, RuleEngine
from boreal_marche.recommend.evaluate import hit_rate
from boreal_marche.utils import git_commit, read_json, utc_now, write_json

MIN_LIFT_GRID = [1.0, 1.5, 2.0, 3.0]


def frequent_itemsets(
    b: Baskets, min_support: float, algo: str, max_len: int
) -> tuple[pd.DataFrame, float]:
    """Mine itemsets; single items below ``min_support`` are pruned first (lossless: no superset of
    an infrequent item can be frequent)."""
    n = b.matrix.shape[0]
    keep = np.asarray(b.matrix.sum(axis=0)).ravel() / n >= min_support
    cols = [i for i, k in zip(b.items, keep, strict=True) if k]
    df = pd.DataFrame(b.matrix[:, keep].toarray(), columns=cols)
    t0 = time.perf_counter()
    if algo == "fpgrowth":
        fi = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=max_len)
    else:
        fi = apriori(
            df, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True
        )
    return fi, time.perf_counter() - t0


def to_rules(itemsets: pd.DataFrame, n_baskets: int, min_confidence: float) -> list[Rule]:
    if itemsets.empty:
        return []
    r = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=n_baskets
    )
    r = r[r["consequents"].map(len) == 1].sort_values(["lift", "confidence"], ascending=False)
    return [
        Rule(i, frozenset(a), next(iter(c)), float(s), float(cf), float(lf))
        for i, (a, c, s, cf, lf) in enumerate(
            zip(
                r["antecedents"],
                r["consequents"],
                r["support"],
                r["confidence"],
                r["lift"],
                strict=True,
            )
        )
    ]


def popularity(b: Baskets, top: int = 200) -> list[str]:
    counts = np.asarray(b.matrix.sum(axis=0)).ravel()
    order = np.argsort(-counts, kind="stable")[:top]
    return [b.items[i] for i in order]


def catalogue(lines: pd.DataFrame) -> dict[str, str]:
    first = lines.drop_duplicates("stock_code")
    return {str(k): str(v) for k, v in zip(first["stock_code"], first["description"], strict=True)}


def sweep(params: Params, paths: Paths) -> dict[str, Any]:
    rp = params.recommend
    lines = load_lines(paths)
    pairs = invoice_items(lines)
    cat = catalogue(lines)
    mine_pairs = pairs[pairs["invoice_date"] < pd.Timestamp(rp.validation_start)]
    val_pairs = pairs[
        (pairs["invoice_date"] >= pd.Timestamp(rp.validation_start))
        & (pairs["invoice_date"] < pd.Timestamp(rp.holdout_start))
    ]
    b = build_matrix(mine_pairs)
    pop = popularity(b)
    rows = []
    for s in rp.min_support_grid:
        fi, secs = frequent_itemsets(b, s, "apriori", rp.max_len)
        for c in rp.min_confidence_grid:
            rules = to_rules(fi, b.matrix.shape[0], c)
            for lift in MIN_LIFT_GRID:
                for rank_by in ("lift", "confidence"):
                    eng = RuleEngine(rules, pop, cat, min_lift=lift, rank_by=rank_by)
                    ev = hit_rate(eng, val_pairs, rp.top_k, params.seed)
                    rows.append(
                        {
                            "min_support": s,
                            "min_confidence": c,
                            "min_lift": lift,
                            "rank_by": rank_by,
                            "n_itemsets": len(fi),
                            "n_rules": len(eng.rules),
                            "apriori_seconds": secs,
                            **ev,
                        }
                    )
            print(f"support={s} conf={c}: {len(rules)} rules (before lift filter)")
    table = pd.DataFrame(rows)
    best = table.sort_values(["hit_rate_hybrid", "n_rules"], ascending=[False, True]).iloc[0]
    payload = {
        "generated_at": utc_now(),
        "mining_period_end": rp.validation_start,
        "validation_slice": [rp.validation_start, rp.holdout_start],
        "n_mining_baskets": int(b.matrix.shape[0]),
        "table": table.to_dict("records"),
        "best_by_hit_rate": best.to_dict(),
    }
    write_json(paths.metrics / "rules_sweep.json", payload)
    print("best:", best.to_dict())
    return payload


def lift_trap(rules: list[Rule], min_lift: float, cat: dict[str, str]) -> dict[str, Any] | None:
    """Highest-confidence rule that the lift filter discards (the 'popular consequent' trap)."""
    below = [r for r in rules if r.lift < min_lift]
    if not below:
        return None
    r = max(below, key=lambda x: x.confidence)
    return {
        "antecedent": sorted(r.antecedent),
        "antecedent_desc": [cat.get(a, "") for a in sorted(r.antecedent)],
        "consequent": r.consequent,
        "consequent_desc": cat.get(r.consequent, ""),
        "support": r.support,
        "confidence": r.confidence,
        "lift": r.lift,
        "consequent_support": r.confidence / r.lift,
    }


def run(params: Params | None = None, paths: Paths | None = None) -> dict[str, Any]:
    params = params or load_params()
    paths = paths or Paths()
    paths.ensure()
    rp = params.recommend
    lines = load_lines(paths)
    pairs = invoice_items(lines)
    cat = catalogue(lines)
    mine_pairs = pairs[pairs["invoice_date"] < pd.Timestamp(rp.holdout_start)]
    b = build_matrix(mine_pairs)
    mem = b.memory()
    fi_fp, t_fp = frequent_itemsets(b, rp.min_support, "fpgrowth", rp.max_len)
    fi_ap, t_ap = frequent_itemsets(b, rp.min_support, "apriori", rp.max_len)
    same = set(map(frozenset, fi_fp["itemsets"])) == set(map(frozenset, fi_ap["itemsets"]))
    if not same:
        raise AssertionError("Apriori and FP-Growth disagree on frequent itemsets")
    all_rules = to_rules(fi_fp, b.matrix.shape[0], rp.min_confidence)
    kept = [r for r in all_rules if r.lift >= rp.min_lift]
    sweep_res = (
        read_json(paths.metrics / "rules_sweep.json")
        if (paths.metrics / "rules_sweep.json").exists()
        else {}
    )
    rank_by = str(sweep_res.get("best_by_hit_rate", {}).get("rank_by", "lift"))
    raw_sha = read_json(paths.manifest)["sha256"]
    header = {
        "mined_at": utc_now(),
        "algorithm": "fpgrowth (mlxtend)",
        "git_commit": git_commit(),
        "source_sha256": raw_sha,
        "mining_period": [str(mine_pairs["invoice_date"].min()), rp.holdout_start],
        "params": {
            "min_support": rp.min_support,
            "min_confidence": rp.min_confidence,
            "min_lift": rp.min_lift,
            "max_len": rp.max_len,
            "rank_by": rank_by,
            "single_item_consequents": True,
        },
        "n_baskets": int(b.matrix.shape[0]),
        "n_items": int(b.matrix.shape[1]),
        "n_frequent_itemsets": int(len(fi_fp)),
        "n_rules_before_lift_filter": len(all_rules),
        "rule_count": len(kept),
    }
    pop = popularity(b)
    payload: dict[str, Any] = {
        "header": header,
        "rules": [
            {
                "id": r.id,
                "antecedent": sorted(r.antecedent),
                "consequent": r.consequent,
                "support": round(r.support, 6),
                "confidence": round(r.confidence, 6),
                "lift": round(r.lift, 4),
            }
            for r in kept
        ],
        "popularity": pop,
        "catalogue": {
            k: cat[k]
            for k in sorted(
                set(pop)
                | {r.consequent for r in kept}
                | {a for r in kept for a in r.antecedent}
                | set(b.items)
            )
            if k in cat
        },
    }
    write_json(paths.rules_file, payload)
    trap = lift_trap(all_rules, rp.min_lift if rp.min_lift > 1 else 2.0, cat)
    top_item = pop[0]
    mining = {
        "generated_at": utc_now(),
        "memory": mem,
        "fpgrowth_seconds": t_fp,
        "apriori_seconds": t_ap,
        "itemsets_identical": same,
        "header": header,
        "lift_trap_example": trap,
        "top_item": {
            "item": top_item,
            "desc": cat.get(top_item, ""),
            "support": float(b.matrix[:, b.items.index(top_item)].sum() / b.matrix.shape[0]),
        },
        "lift_distribution": {
            q: float(np.quantile([r.lift for r in all_rules], q))
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        }
        if all_rules
        else {},
        "example_rules": payload["rules"][:5],
    }
    write_json(paths.metrics / "rules_mining.json", mining)
    tracking.setup(params, paths, params.mlflow.experiment_recommend)
    with mlflow.start_run(
        run_name="mine_rules", tags={"git_commit": git_commit(), "raw_sha256": raw_sha}
    ):
        mlflow.log_params(header["params"])
        mlflow.log_metrics(
            {
                "n_rules": len(kept),
                "n_itemsets": len(fi_fp),
                "fpgrowth_seconds": t_fp,
                "apriori_seconds": t_ap,
                "dense_mb": mem["dense_bool_mb"],
                "sparse_mb": mem["sparse_csr_mb"],
            }
        )
        mlflow.log_artifact(str(paths.rules_file))
    print(
        f"{len(kept)} rules (of {len(all_rules)}) -> {paths.rules_file}; "
        f"FP-Growth {t_fp:.1f}s, Apriori {t_ap:.1f}s; memory {mem}"
    )
    return mining


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="mine", choices=["mine", "sweep"])
    if ap.parse_args().mode == "sweep":
        sweep(load_params(), Paths())
    else:
        run()
