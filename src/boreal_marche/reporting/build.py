"""Render D1-D3 from templates: ``reports/src/*.md`` -> ``reports/977_D<n>_*.md|html|pdf``.

Templates contain ``{{ key }}`` placeholders. Every value is computed here from pipeline outputs
(``reports/metrics/*.json``, artifacts, params) so a number in a report cannot drift from the code
that produced it (Appendix A: "Every number in the text matches the number in the code"). An unknown
key fails the build.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import markdown  # type: ignore[import-untyped]
import pandas as pd

from boreal_marche.config import Paths, load_params
from boreal_marche.features import FEATURE_SPEC
from boreal_marche.utils import read_json

REPORTS = {
    "D1": ("D1_Project_Proposal_Charter.md", "977_D1_Morgunova_ProjectProposal"),
    "D2": ("D2_Design_Architecture.md", "977_D2_Morgunova_DesignArchitecture"),
    "D3": ("D3_Implementation_Validation.md", "977_D3_Morgunova_ImplementationValidation"),
}
FAMILY_RATIONALE = {
    "recency": "Time since the last order is the classical strongest churn signal (the R of RFM).",
    "frequency": "Habit: customers who order often, and in many distinct months, keep ordering.",
    "monetary": "Value and order size separate small retail buyers from wholesale accounts, "
    "whose re-order behaviour differs.",
    "cadence": "A customer's own re-order cycle: being late against it (overdue ratio) or slowing "
    "down (trend) signals lapse better than a global recency cut-off.",
    "basket": "Broad, replenishment-style baskets indicate an established supplier relationship.",
    "category": "Seasonal-only buyers have long, predictable gaps; diversified buyers are "
    "stickier.",
    "behaviour": "Cancellations signal dissatisfaction; order timing separates business from "
    "consumer buying (and is a minimisation candidate).",
    "geography": "Delivery region affects service levels; also a fairness-audit subgroup.",
    "context": "The label base rate is seasonal (Q4 peak) and the quarter is known at prediction "
    "time.",
}
PRETTY = {
    "baseline_dummy_prior": "Baseline: Dummy (prior)",
    "baseline_rule_recency60": "Baseline: rule 'ordered in last 60 days'",
    "baseline_recency_deciles": "Heuristic: recency deciles",
    "logreg": "Logistic regression",
    "gaussian_nb": "Gaussian naive Bayes",
    "knn": "k-NN (k=50)",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "hist_gb": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "mlp": "MLP (64-32)",
}


def f3(x: float) -> str:
    return f"{x:.3f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


def n0(x: float) -> str:
    return f"{x:,.0f}"


def money(x: float) -> str:
    return f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def md_table(header: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def mlflow_counts(paths: Paths) -> dict[str, int]:
    import mlflow
    from mlflow.tracking import MlflowClient

    from boreal_marche.tracking import tracking_uri

    mlflow.set_tracking_uri(tracking_uri(load_params(), paths))
    c = MlflowClient()
    counts: dict[str, int] = {}
    for e in c.search_experiments():
        if e.name == "Default":
            continue
        counts[e.name] = len(c.search_runs([e.experiment_id], max_results=5000))
    for r in c.search_runs(
        [e.experiment_id for e in c.search_experiments()], filter_string="tags.stage = 'tuning'"
    ):
        name = r.data.tags.get("model_family", "?")
        counts[f"tune_wall_{name}"] = int((r.info.end_time - r.info.start_time) / 1000)
    counts["registry_versions"] = len(
        c.search_model_versions(f"name='{load_params().mlflow.registered_model}'")
    )
    return counts


def context(paths: Paths) -> dict[str, str]:  # noqa: PLR0915 - one flat, auditable mapping
    p = load_params()
    m = paths.metrics
    man = read_json(paths.manifest)
    clean = read_json(m / "cleaning_report.json")
    gold = read_json(m / "gold_summary.json")
    spot = read_json(m / "spotcheck.json")
    tune = read_json(m / "tuning.json")
    fin = read_json(m / "train_final.json")
    test = read_json(m / "test_evaluation.json")
    look1 = read_json(m / "test_evaluation_look1.json")
    leak = read_json(m / "leakage.json")
    sweep = read_json(m / "rules_sweep.json")
    mine = read_json(m / "rules_mining.json")
    rev = read_json(m / "rules_evaluation.json")
    meta = read_json(paths.model_meta)
    runs = mlflow_counts(paths)
    c: dict[str, str] = {}

    # --- data -----------------------------------------------------------------------------
    rules = {r["rule"].split()[0]: r for r in clean["rules"]}
    b, s = clean["bronze"], clean["silver"]
    sheets = list(man["sheet_row_counts"].items())
    c |= {
        "raw_rows": n0(man["row_count"]),
        "sheet1": f"{sheets[0][0]} ({n0(sheets[0][1])})",
        "sheet2": f"{sheets[1][0]} ({n0(sheets[1][1])})",
        "sha256": man["sha256"],
        "sha_short": man["sha256"][:12],
        "raw_mb": f"{man['raw_bytes'] / 2**20:.1f}",
        "retrieved": man["retrieved_at"][:10],
        "date_min": b["date_min"][:10],
        "date_max": b["date_max"][:10],
        "bronze_invoices": n0(b["invoices"]),
        "bronze_skus": n0(b["skus"]),
        "bronze_customers": n0(b["customers"]),
        "silver_lines": n0(s["lines"]),
        "silver_invoices": n0(s["invoices"]),
        "silver_skus": n0(s["skus"]),
        "silver_customers": n0(s["customers"]),
        "head_a_lines": n0(s["head_a_lines"]),
        "head_a_invoices": n0(s["head_a_invoices"]),
        "lines_per_invoice": f"{s['mean_lines_per_invoice']:.1f}",
        "silver_revenue": n0(s["revenue"]),
        "removed_total": n0(b["lines"] - s["lines"]),
        "removed_pct": pct((b["lines"] - s["lines"]) / b["lines"]),
        "overlap_rows": n0(rules["R1"]["rows_removed"]),
        "dup_rows": n0(rules["R2"]["rows_removed"]),
        "cancel_rows": n0(rules["R3"]["rows_removed"]),
        "adj_rows": n0(rules["R4"]["rows_removed"]),
        "nonpos_qty": n0(rules["R5"]["rows_removed"]),
        "nonpos_price": n0(rules["R6"]["rows_removed"]),
        "service_rows": n0(rules["R7"]["rows_removed"]),
        "desc_changed": n0(rules["R8"]["rows_flagged"]),
        "outliers": n0(rules["R9"]["rows_flagged"]),
        "qty_cap": n0(clean["extras"]["qty_cap"]),
        "rev_cap": f"{clean['extras']['rev_cap']:,.2f}",
        "guest_rows": n0(rules["R10"]["rows_flagged"]),
        "guest_pct": pct(rules["R10"]["rows_flagged"] / s["lines"]),
        "desc_before": n0(clean["extras"]["distinct_descriptions_before"]),
        "desc_after": n0(clean["extras"]["distinct_descriptions_after"]),
    }
    c["cleaning_table"] = md_table(
        ["Rule", "Rows before", "Rows after", "Removed", "Flagged", "Why the rule exists"],
        [
            [
                r["rule"],
                n0(r["rows_before"]),
                n0(r["rows_after"]),
                n0(r["rows_removed"]),
                n0(r["rows_flagged"]),
                r["rationale"],
            ]
            for r in clean["rules"]
        ],
        "lrrrrl",
    )

    # --- gold / features --------------------------------------------------------------------
    snaps = gold["snapshots"]
    c["n_features"] = str(gold["n_features"])
    c["n_snapshots_rows"] = n0(sum(x["customers"] for x in snaps))
    c["snapshot_table"] = md_table(
        ["Split", "Cutoff T", "Eligible customers", "Returned within 90 d"],
        [
            [x["split"], x["cutoff"][:10], n0(x["customers"]), pct(x["positive_rate"])]
            for x in sorted(snaps, key=lambda z: z["cutoff"])
        ],
    )
    miss = gold["missing_share"]
    c["missing_gap"] = pct(miss["mean_gap_days"])
    c["missing_std"] = pct(miss["std_gap_days"])
    c["feature_table"] = md_table(
        ["#", "Feature", "Family", "Type", "Definition", "Missingness"],
        [
            [i, f"`{f.name}`", f.family, f.dtype, f.definition, f.missing]
            for i, f in enumerate(FEATURE_SPEC, 1)
        ],
        "rlllll",
    )
    c["feature_rationale_table"] = md_table(
        ["Family", "Features", "Why it should predict a 90-day return"],
        [
            [
                fam,
                ", ".join(f"`{f.name}`" for f in FEATURE_SPEC if f.family == fam),
                FAMILY_RATIONALE[fam],
            ]
            for fam in dict.fromkeys(f.family for f in FEATURE_SPEC)
        ],
        "lll",
    )

    # --- spot-check ----------------------------------------------------------------------
    tbl = sorted(spot["table"], key=lambda r: -r["pr_auc_mean"])
    c["spot_rows"] = n0(spot["n_rows"])
    c["spot_pos"] = pct(spot["positive_rate"])
    c["spot_table"] = md_table(
        ["Model", "PR-AUC (mean ± sd)", "ROC-AUC", "Brier", "Recall@top10%", "Fit s"],
        [
            [
                PRETTY.get(r["model"], r["model"]),
                f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
                f3(r["roc_auc_mean"]),
                f3(r["brier_mean"]),
                f3(r["recall_top_decile_mean"]),
                f"{r['fit_seconds_mean']:.2f}",
            ]
            for r in tbl
        ],
    )
    by = {r["model"]: r for r in spot["table"]}
    top = [r for r in tbl if not r["model"].startswith("baseline_")]
    c["spot_best"] = PRETTY[top[0]["model"]]
    c["spot_best_score"] = f4(top[0]["pr_auc_mean"])
    c["spot_second"] = PRETTY[top[1]["model"]]
    c["spot_second_score"] = f4(top[1]["pr_auc_mean"])
    c["spot_third"] = PRETTY[top[2]["model"]]
    c["spot_third_score"] = f4(top[2]["pr_auc_mean"])
    c["spot_worst"] = PRETTY[top[-1]["model"]]
    c["spot_worst_score"] = f4(top[-1]["pr_auc_mean"])
    c["spot_top_sd"] = f4(top[0]["pr_auc_std"])
    c["spot_gap_12"] = f4(top[0]["pr_auc_mean"] - top[1]["pr_auc_mean"])
    within = [r for r in top if r["pr_auc_mean"] >= top[0]["pr_auc_mean"] - top[0]["pr_auc_std"]]
    c["spot_within_sd"] = str(len(within))
    for k in ("baseline_dummy_prior", "baseline_rule_recency60", "baseline_recency_deciles"):
        c[f"cv_{k}"] = f4(by[k]["pr_auc_mean"])
    c["top2"] = " and ".join(PRETTY[x] for x in spot["top2"])

    # --- tuning ----------------------------------------------------------------------------
    rows = []
    for name, r in tune["results"].items():
        imp = sorted(r["importances"].items(), key=lambda kv: -kv[1])[:3]
        rows.append(
            [
                PRETTY[name],
                r["n_trials"],
                r["n_complete"],
                r["n_pruned"],
                f4(r["default_value"]),
                f4(r["best_value"]),
                f"{r['improvement']:+.4f}",
                ", ".join(f"{k} ({v:.2f})" for k, v in imp),
            ]
        )
        c[f"tune_{name}_best"] = f4(r["best_value"])
        c[f"tune_{name}_impr"] = f"{r['improvement']:+.4f}"
        c[f"tune_{name}_pruned"] = str(r["n_pruned"])
        c[f"tune_{name}_params"] = ", ".join(
            f"`{k}`={v:.4g}" if isinstance(v, float) else f"`{k}`={v}"
            for k, v in r["best_params"].items()
        )
    c["tuning_table"] = md_table(
        [
            "Model",
            "Trials",
            "Complete",
            "Pruned",
            "Default CV PR-AUC",
            "Best CV PR-AUC",
            "Gain",
            "Top-3 hyper-parameter importances",
        ],
        rows,
        "lrrrrrrl",
    )
    c["tune_winner"] = PRETTY[tune["winner"]]
    c["n_trials_total"] = str(sum(r["n_trials"] for r in tune["results"].values()))
    c["n_pruned_total"] = str(sum(r["n_pruned"] for r in tune["results"].values()))

    # --- calibration / threshold ---------------------------------------------------------
    champ = fin["champion"]
    cand = fin["candidates"][champ]
    cal = cand["calibration"]
    c["champion"] = PRETTY[champ]
    c["calib_method"] = cand["method"]
    c["calibration_table"] = md_table(
        ["Method (5-fold out-of-fold on validation)", "Brier", "Log-loss", "ECE"],
        [[k, f4(v["brier"]), f4(v["log_loss"]), f3(v["ece"])] for k, v in cal.items()],
    )
    c["ece_raw"], c["ece_cal"] = f3(cal["none"]["ece"]), f3(cal[cand["method"]]["ece"])
    c["brier_raw"], c["brier_cal"] = f4(cal["none"]["brier"]), f4(cal[cand["method"]]["brier"])
    c["val_pr_auc"] = f4(cand["val_metrics_cal"]["pr_auc"])
    c["val_roc_auc"] = f4(cand["val_metrics_cal"]["roc_auc"])
    bv = fin["baselines_validation"]
    c["val_best_baseline"] = f4(max(v["pr_auc"] for v in bv.values()))
    c["val_dummy"] = f4(bv["baseline_dummy_prior"]["pr_auc"])
    c["val_rule"] = f4(bv["baseline_rule_recency60"]["pr_auc"])
    c["val_deciles"] = f4(bv["baseline_recency_deciles"]["pr_auc"])
    other = [k for k in fin["candidates"] if k != champ][0]
    c["challenger"] = PRETTY[other]
    c["challenger_val_pr_auc"] = f4(fin["candidates"][other]["val_metrics_cal"]["pr_auc"])
    promo = cand["promotion"]["checks"]
    c["promotion_table"] = md_table(
        ["Gate", "Result"],
        [[k.replace("_", " "), "pass" if v else "FAIL"] for k, v in promo.items()],
        "ll",
    )
    vp = fin["validation_policy"]
    c["threshold"] = f"{fin['threshold']:.2f}"
    c |= {
        "val_n": n0(vp["n_customers"]),
        "val_contacted": n0(vp["model_contacted"]),
        "val_contacted_pct": pct(vp["model_contacted"] / vp["n_customers"]),
        "val_not_sent_pct": pct(1 - vp["model_contacted"] / vp["n_customers"], 0),
        "val_profit": money(vp["model_profit"]),
        "val_everyone": money(vp["everyone_profit"]),
        "val_gain": money(vp["model_vs_everyone"]),
        "val_ev_profit": money(vp["ev_policy_profit"]),
        "val_ev_contacted": n0(vp["ev_policy_contacted"]),
        "val_lapsers": n0(vp["lapsers_total"]),
        "val_lapsers_hit": n0(vp["lapsers_contacted_model"]),
        "val_profit_1000": money(vp["model_profit_per_1000"]),
        "val_everyone_1000": money(vp["everyone_profit_per_1000"]),
    }
    c["sensitivity_table"] = md_table(
        ["Assumption", "Profit-maximising threshold", "Max validation profit"],
        [
            [
                k.replace("save_rate=", "save rate = "),
                f"{v['best_threshold']:.2f}",
                money(v["max_profit"]),
            ]
            for k, v in fin["threshold_sensitivity"].items()
        ]
        + [
            [
                "save rate 0.20, value horizon 90 d (first design)",
                f"{fin['horizon_90_negative_result']['best_threshold']:.2f}",
                money(fin["horizon_90_negative_result"]["max_profit"]),
            ]
        ],
    )
    c["h90_everyone"] = money(fin["horizon_90_negative_result"]["everyone_profit"])
    bp = p.business
    c |= {
        "coupon_cost": f"${bp.coupon_cost:.2f}",
        "discount_rate": pct(bp.discount_rate, 0),
        "save_rate": f"{bp.save_rate:.2f}",
        "gross_margin": pct(bp.gross_margin, 0),
        "horizon": str(bp.value_horizon_days),
    }

    # --- test ---------------------------------------------------------------------------------
    tm, ci = test["metrics"], test["ci95"]
    c |= {
        "test_n": n0(test["n"]),
        "test_pos": pct(test["positive_rate"]),
        "test_looks": str(test["n_test_looks"]),
        "test_pr_auc": f4(tm["pr_auc"]),
        "test_pr_ci": f"[{ci['pr_auc']['lo']:.4f}, {ci['pr_auc']['hi']:.4f}]",
        "test_roc": f4(tm["roc_auc"]),
        "test_brier": f4(tm["brier"]),
        "test_ece": f3(tm["ece"]),
        "test_pr_lapse": f4(tm["pr_auc_lapse"]),
        "look1_pr_auc": f4(look1["metrics"]["pr_auc"]),
    }
    trows = [
        [
            "Model (" + c["champion"] + ", calibrated)",
            f4(tm["pr_auc"]),
            f"{ci['pr_auc']['lo']:.3f}–{ci['pr_auc']['hi']:.3f}",
            f4(tm["roc_auc"]),
            f4(tm["brier"]),
            f3(tm["recall_top_decile"]),
        ]
    ]
    for k, v in test["baselines"].items():
        trows.append(
            [
                PRETTY[k],
                f4(v["pr_auc"]),
                f"{v['ci']['pr_auc']['lo']:.3f}–{v['ci']['pr_auc']['hi']:.3f}",
                f4(v["roc_auc"]),
                f4(v["brier"]),
                f3(v["recall_top_decile"]),
            ]
        )
    c["test_table"] = md_table(
        ["", "PR-AUC", "95% CI (bootstrap)", "ROC-AUC", "Brier", "Recall@top10%"], trows
    )
    best_base_test = max(v["pr_auc"] for v in test["baselines"].values())
    c["test_gain_vs_base"] = f"{tm['pr_auc'] - best_base_test:+.4f}"
    c["test_best_base"] = f4(best_base_test)
    tp = test["policy"]
    c |= {
        "test_contacted": n0(tp["model_contacted"]),
        "test_profit": money(tp["model_profit"]),
        "test_everyone": money(tp["everyone_profit"]),
        "test_gain": money(tp["model_vs_everyone"]),
        "test_ev_profit": money(tp["ev_policy_profit"]),
        "test_ev_contacted": n0(tp["ev_policy_contacted"]),
        "test_lapsers": n0(tp["lapsers_total"]),
        "test_lapsers_hit": n0(tp["lapsers_contacted_model"]),
        "annual_gain": money(tp["model_vs_everyone"] * 4),
    }
    seg = [x for x in test["segments"] if x["n"] >= 50]

    def opt(v: float | None, fmt: Any = f3) -> str:
        return "n/a" if v is None else fmt(v)

    c["segment_table"] = md_table(
        [
            "Segment",
            "n",
            "Return rate",
            "PR-AUC",
            "Lapser recall",
            "Selection rate",
            "Recall gap vs overall",
        ],
        [
            [
                f"{x['segment'].replace('_', ' ')}: {x['value']}",
                n0(x["n"]),
                pct(x["return_rate"]),
                opt(x["pr_auc"]),
                opt(x["lapse_recall"]),
                pct(x["selection_rate"]),
                opt(x["recall_gap_vs_overall"], lambda v: f"{100 * v:+.1f} pts"),
            ]
            for x in seg
        ],
    )
    segd = {(x["segment"], x["value"]): x for x in test["segments"]}
    for key, (sg, val) in {
        "q1": ("spend_quartile", "Q1 (low)"),
        "q4": ("spend_quartile", "Q4 (high)"),
        "one": ("frequency_band", "1 order"),
        "four": ("frequency_band", "4+ orders"),
        "new": ("tenure_band", "<6 months"),
    }.items():
        x = segd[(sg, val)]
        c[f"seg_{key}_pr"] = opt(x["pr_auc"])
        c[f"seg_{key}_recall"] = opt(x["lapse_recall"])
        c[f"seg_{key}_n"] = n0(x["n"])

    # --- leakage ---------------------------------------------------------------------------
    lw, sl = leak["label_window_leak"], leak["split_leak"]
    c |= {
        "leak_clean": f4(lw["clean"]["pr_auc"]),
        "leak_leaky": f4(lw["leaky"]["pr_auc"]),
        "leak_clean_roc": f4(lw["clean"]["roc_auc"]),
        "leak_leaky_roc": f4(lw["leaky"]["roc_auc"]),
        "split_row": f4(sl["row_kfold"]["pr_auc_mean"]),
        "split_group": f4(sl["grouped_kfold"]["pr_auc_mean"]),
        "split_temporal": f4(sl["temporal_validation"]["pr_auc"]),
        "split_shared": pct(
            float(
                pd.Series(sl["share_of_val_rows_whose_customer_is_in_train_fold_rowkfold"]).mean()
            ),
            0,
        ),
    }
    c["netting_table"] = md_table(
        ["Cutoff", "Pre-cutoff lines netting would touch", "Customers affected"],
        [
            [
                x["cutoff"],
                n0(x["pre_cutoff_lines_that_netting_would_touch"]),
                n0(x["customers_affected"]),
            ]
            for x in leak["netting_leak"]
        ],
    )

    # --- head B -------------------------------------------------------------------------------
    mem, hdr, trap = mine["memory"], mine["header"], mine["lift_trap_example"]
    c |= {
        "b_baskets": n0(mem["n_baskets"]),
        "b_items": n0(mem["n_items"]),
        "b_density": pct(mem["density"], 2),
        "b_dense_mb": f"{mem['dense_bool_mb']:.1f}",
        "b_sparse_mb": f"{mem['sparse_csr_mb']:.1f}",
        "b_ratio": f"{mem['ratio']:.1f}",
        "b_fp_s": f"{mine['fpgrowth_seconds']:.1f}",
        "b_ap_s": f"{mine['apriori_seconds']:.1f}",
        "b_speedup": f"{mine['fpgrowth_seconds'] / mine['apriori_seconds']:.0f}",
        "b_itemsets": n0(hdr["n_frequent_itemsets"]),
        "b_rules_before": n0(hdr["n_rules_before_lift_filter"]),
        "b_rules": n0(hdr["rule_count"]),
        "b_support": f"{hdr['params']['min_support']:g}",
        "b_support_invoices": n0(hdr["params"]["min_support"] * mem["n_baskets"]),
        "b_conf": f"{hdr['params']['min_confidence']:g}",
        "b_lift": f"{hdr['params']['min_lift']:g}",
        "b_rank": hdr["params"]["rank_by"],
        "b_top_item": f"{mine['top_item']['item']} {mine['top_item']['desc']}",
        "b_top_support": pct(mine["top_item"]["support"]),
        "trap_ante": ", ".join(
            f"{a} {d}" for a, d in zip(trap["antecedent"], trap["antecedent_desc"], strict=True)
        ),
        "trap_cons": f"{trap['consequent']} {trap['consequent_desc']}",
        "trap_conf": pct(trap["confidence"]),
        "trap_lift": f"{trap['lift']:.3f}",
        "trap_support": pct(trap["support"], 2),
        "trap_cons_support": pct(trap["consequent_support"]),
        "lift_median": f"{mine['lift_distribution']['0.5']:.1f}",
        "hold_cases": n0(rev["n_cases"]),
        "hold_rules": f3(rev["hit_rate_rules_only"]),
        "hold_hybrid": f3(rev["hit_rate_hybrid"]),
        "hold_pop": f3(rev["hit_rate_popularity"]),
        "hold_ratio": f"{rev['hybrid_vs_popularity_ratio']:.1f}",
        "hold_cov": pct(rev["basket_coverage"]),
        "hold_cat_cov": pct(rev["catalogue_coverage"]),
        "hold_items": n0(rev["distinct_items_recommended"]),
        "val_slice_cases": n0(sweep["table"][0]["n_cases"]),
    }
    t = pd.DataFrame(sweep["table"])
    sub = t[
        (t["min_confidence"] == p.recommend.min_confidence)
        & (t["min_lift"] == p.recommend.min_lift)
        & (t["rank_by"] == hdr["params"]["rank_by"])
    ].sort_values("min_support")
    c["sweep_table"] = md_table(
        [
            "min_support",
            "Itemsets",
            "Rules",
            "Apriori s",
            "Hit@5 rules-only",
            "Hit@5 hybrid",
            "Hit@5 popularity",
            "Basket coverage",
        ],
        [
            [
                f"{r.min_support:g}",
                n0(r.n_itemsets),
                n0(r.n_rules),
                f"{r.apriori_seconds:.2f}",
                f3(r.hit_rate_rules_only),
                f3(r.hit_rate_hybrid),
                f3(r.hit_rate_popularity),
                pct(r.basket_coverage),
            ]
            for r in sub.itertuples()
        ],
    )
    lifts = t[
        (t["min_support"] == p.recommend.min_support)
        & (t["min_confidence"] == p.recommend.min_confidence)
    ]
    c["lift_rank_table"] = md_table(
        ["min_lift", "Rank by", "Rules", "Hit@5 hybrid"],
        [
            [f"{r['min_lift']:g}", r["rank_by"], n0(r["n_rules"]), f3(r["hit_rate_hybrid"])]
            for r in lifts.sort_values(["rank_by", "min_lift"]).to_dict("records")
        ],
    )
    ex = rev["strategies"]["matching_basket"]
    c["example_basket"] = ", ".join(ex["basket"])
    c["example_recs"] = "; ".join(
        f"{r['item']} {r['description']} ({r['source']}"
        + (f", lift {r['lift']:.0f}" if r["lift"] else "")
        + ")"
        for r in ex["recommendations"][:3]
    )

    # --- MLOps -----------------------------------------------------------------------------
    c |= {
        "mlflow_runs_retention": n0(runs.get(p.mlflow.experiment_retention, 0)),
        "mlflow_runs_recommend": n0(runs.get(p.mlflow.experiment_recommend, 0)),
        "registry_versions": str(runs["registry_versions"]),
        **{f"tune_wall_{k}": f"{runs.get(f'tune_wall_{k}', 0) / 60:.1f}" for k in tune["results"]},
        "model_version": meta["model_version"],
        "model_run_id": meta["mlflow_run_id"][:12],
        "model_git": meta["git_commit"][:12]
        + ("…-dirty" if meta["git_commit"].endswith("-dirty") else "…"),
        "model_n_rows": n0(meta["n_train_rows"]),
        "model_kb": n0(paths.model_file.stat().st_size / 1024),
        "rules_kb": n0(paths.rules_file.stat().st_size / 1024),
        "sklearn_v": meta["library_versions"]["scikit-learn"],
        "xgb_v": meta["library_versions"]["xgboost"],
        "py_v": meta["library_versions"]["python"],
    }
    c |= {
        "obs_days": str(p.windows.observation_days),
        "label_days": str(p.windows.label_days),
        "train_cutoffs": ", ".join(p.windows.cutoffs.train),
        "val_cutoff": p.windows.cutoffs.validation[0],
        "test_cutoff": p.windows.cutoffs.test[0],
        "n_folds": str(p.spotcheck.n_folds),
        "seed": str(p.seed),
        "trials_per_model": str(p.tuning.n_trials),
    }
    return c


def render(text: str, ctx: dict[str, str]) -> str:
    def rep(mt: re.Match[str]) -> str:
        key = mt.group(1)
        if key not in ctx:
            raise KeyError(f"unknown template key {key!r}")
        return ctx[key]

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", rep, text)


CSS = """
@page { size: Letter; margin: 18mm 17mm 18mm 17mm;
  @bottom-center { content: counter(page); } }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38;
  color: #0b0b0b; }
h1 { font-size: 17pt; color: #184f95; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; color: #184f95; border-bottom: 1px solid #cde2fb; padding-bottom: 2pt;
  margin: 14pt 0 5pt; page-break-after: avoid; }
h3 { font-size: 10.5pt; color: #0d366b; margin: 10pt 0 3pt; page-break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8pt;
  page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #e8f0fb; color: #0d366b; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d6d5d0; padding: 2.5pt 4pt; vertical-align: top; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; background: #f4f3ef;
  padding: 0 2px; }
pre { background: #f4f3ef; padding: 5pt; font-size: 7.8pt; white-space: pre-wrap; }
img { max-width: 100%; display: block; margin: 4pt auto 2pt; page-break-inside: avoid; }
.caption, p em:only-child { font-size: 8pt; color: #52514e; }
blockquote { border-left: 3px solid #eb6834; margin: 6pt 0; padding: 2pt 8pt; background: #fdf5f1; }
.title { border-bottom: 2px solid #2a78d6; margin-bottom: 8pt; padding-bottom: 6pt; }
.title p { margin: 1pt 0; color: #52514e; }
.pagebreak { page-break-before: always; }
"""


def find_browser() -> str:
    """Chromium-based browser for headless PDF printing: CHROME_PATH, PATH, then the usual
    install locations (Windows Chrome/Edge — Edge ships with every Windows 10/11 — and macOS)."""
    import os

    env = os.environ.get("CHROME_PATH")
    if env and Path(env).exists():
        return env
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        Path(os.environ.get(var, "")) / sub
        for var in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
        for sub in (
            r"Google\Chrome\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe",
        )
        if os.environ.get(var)
    ] + [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError("No Chrome/Chromium/Edge found for PDF output; set CHROME_PATH")


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        [
            find_browser(),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )


def run() -> None:
    paths = Paths()
    src = paths.reports / "src"
    ctx = context(paths)
    from pypdf import PdfReader

    for code, (tpl, out) in REPORTS.items():
        text = render((src / tpl).read_text(encoding="utf-8"), ctx)
        md_path = paths.reports / f"{out}.md"
        md_path.write_text(text, encoding="utf-8")
        body = markdown.markdown(
            text, extensions=["tables", "fenced_code", "attr_list", "md_in_html", "sane_lists"]
        )
        html = (
            f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{code}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>"
        )
        html_path = paths.reports / f"{out}.html"
        html_path.write_text(html, encoding="utf-8")
        pdf_path = paths.reports / f"{out}.pdf"
        to_pdf(html_path, pdf_path)
        print(f"{code}: {pdf_path.name} — {len(PdfReader(pdf_path).pages)} pages")


if __name__ == "__main__":
    run()
