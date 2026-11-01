"""D5 template values and figures, computed from the saved evidence (never typed by hand).

Sources: ``latency_profile.json`` (scripts/latency_profile.py), ``scalability.json``
(scripts/scalability_probe.py), ``interpretability.json``, ``fairness.json`` and
``fairness_v09_37features.json``, ``minimisation.json``, ``retrain_decision.json``, the test
evaluations (look 2 = v0.9 model, current = look 3), ``assistant_tokens.json`` and, once collected,
``aws_costs.json`` (scripts/collect_aws_costs.py). Prices are the published list prices stated in
``PRICES`` with the date they were read; missing evidence renders as PENDING.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from boreal_marche.config import Paths, load_params
from boreal_marche.features import EXCLUDED_FROM_MODEL, FEATURES, MODEL_FEATURES
from boreal_marche.reporting.style import BLUE, INK_2, NEUTRAL, ORANGE
from boreal_marche.utils import read_json

PENDING = "**PENDING**"
# Published list prices, read 2026-11-01 (us-east-1). App Runner: aws.amazon.com/apprunner/pricing;
# g4dn.xlarge: instances.vantage.sh (the official EC2 page is dynamic). The OpenAI price is the one
# in params.yaml; openai.com refused the automated read, so it is marked unverified in the report.
PRICES = {
    "apprunner_vcpu_h": 0.064,
    "apprunner_gb_h": 0.007,
    "g4dn_xlarge_h": 0.526,
    "ecr_gb_month": 0.10,
}
HOURS_PER_MONTH = 730
VCPU, GB = 1, 2  # the App Runner service configuration (D4 §6)
CPU_DERATE = 0.5  # one App Runner vCPU vs one core of the measuring desktop (assumption, stated)
LATENCY_ROWS = [
    "/predict",
    "/predict/batch (100)",
    "/products",
    "/recommend",
    "/chat policy (rule)",
    "/chat retention tool (rule)",
    "/monitoring/report",
]


def _md(header: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _opt(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def _pct(x: float | None, d: int = 0) -> str:
    return "—" if x is None else f"{100 * x:.{d}f}%"


def _money(x: float, d: int = 0) -> str:
    return f"−${abs(x):,.{d}f}" if x < 0 else f"${x:,.{d}f}"


def _ms(x: float) -> str:
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


# ------------------------------------------------------------------------------- figures ---


def latency_figure(paths: Paths, before: dict, after: dict) -> None:
    rows = [r for r in LATENCY_ROWS if r in before["warm_client_ms"]]
    y = np.arange(len(rows))[::-1]
    b = [before["warm_client_ms"][r]["p50"] for r in rows]
    a = [after["warm_client_ms"][r]["p50"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.barh(y + 0.2, b, 0.38, color=NEUTRAL, label="before (v0.9)")
    ax.barh(y - 0.2, a, 0.38, color=BLUE, label="after (v0.9.5)")
    for yi, v in zip(y, a, strict=True):
        ax.text(v * 1.08, yi - 0.2, f"{_ms(v)} ms", va="center", fontsize=7, color=INK_2)
    ax.set_xscale("log")
    ax.set_yticks(y, rows)
    ax.set_xlabel("warm latency p50, client side, ms (log scale)")
    ax.set_title("API latency by endpoint, before and after the D5 changes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(paths.figures / "d5_latency.png", dpi=160)
    plt.close(fig)


def importance_figure(paths: Paths, interp: dict) -> None:
    g = [x for x in interp["global"] if x["mean_abs_effect"] > 0][:12]
    names = [x["feature"] for x in g][::-1]
    y = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    a1.barh(y, [x["mean_abs_effect"] for x in g][::-1], color=BLUE)
    a1.set_title("Mean |contribution| (log-odds)", fontsize=9)
    a2.barh(y, [x["perm_pr_auc_drop"] for x in g][::-1], color=BLUE)
    a2.set_title("Permutation: PR-AUC drop", fontsize=9)
    a1.set_yticks(y, names)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Global importance, validation snapshot (19-feature model)", fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(paths.figures / "d5_importance.png", dpi=160)
    plt.close(fig)


def fairness_figure(paths: Paths, fair: dict) -> None:
    pols = list(fair["policies"]["validation"])
    quarts = ["Q1 (low)", "Q2", "Q3", "Q4 (top)"]
    fig, axes = plt.subplots(1, len(pols), figsize=(7.6, 2.8), sharey=True)
    x = np.arange(len(quarts))
    for ax, name in zip(axes, pols, strict=True):
        for k, (snap, col) in enumerate((("validation", BLUE), ("test", ORANGE))):
            s = fair["policies"][snap][name]
            vals = [s["recall_by_spend"].get(q) or 0 for q in quarts]
            ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=col, label=snap)
            ax.axhline(s["lapser_recall"], color=col, ls="--", lw=1)
        ax.set_xticks(x, ["Q1", "Q2", "Q3", "Q4"])
        ax.set_title(name.split("_", 1)[1].replace("_", " "), fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("lapser recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0))
    fig.suptitle(
        "Lapser recall by spend quartile per policy (dashed: overall)",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(paths.figures / "d5_fairness.png", dpi=160)
    plt.close(fig)


def scalability_figure(paths: Paths, sc: dict) -> None:
    panels = [
        ("rules", "n_rules", "p95", "rules served", "/recommend p95 (ms)"),
        ("retrieval", "n_chunks", "ms", "indexed chunks", "search (ms)"),
        ("logstore", "window", "report_ms", "predictions in window", "report (ms)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.5))
    for ax, (key, xk, yk, xl, yl) in zip(axes, panels, strict=True):
        rows = sc[key]["rows"] if key != "logstore" else sc[key]["report"]
        xs, ys = [r[xk] for r in rows], [max(r[yk], 1e-3) for r in rows]
        ax.plot(xs, ys, color=BLUE, marker="o", ms=4)
        ax.axhline(50, color=NEUTRAL, ls="--", lw=1)
        ax.text(xs[0], 56, "50 ms", color=INK_2, fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    fig.suptitle(
        "Measured growth of the three components that scale with data",
        fontsize=10,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(paths.figures / "d5_scalability.png", dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------- context ---


def _latency(c: dict[str, str], lp: dict) -> None:
    R = lp["runs"]
    before, code, after = R["before"], R["after_code"], R["after"]
    rows = []
    for r in LATENCY_ROWS:
        b, a = before["warm_client_ms"][r], after["warm_client_ms"][r]
        bs = before["warm_server_ms"].get(r.split(" ")[0])
        as_ = after["warm_server_ms"].get(r.split(" ")[0])
        rows.append(
            [
                f"`{r}`",
                f"{_ms(before['first_call_ms'][r])} / {_ms(after['first_call_ms'][r])}",
                f"{_ms(b['p50'])} / {_ms(b['p95'])}",
                f"{_ms(a['p50'])} / {_ms(a['p95'])}",
                f"{_ms(bs['p50'])} → {_ms(as_['p50'])}" if bs and as_ and "chat" not in r else "—",
                _pct(1 - a["p50"] / b["p50"]),
            ]
        )
    c["latency_table"] = _md(
        [
            "Endpoint",
            "First call before / after",
            "Warm p50 / p95 before",
            "Warm p50 / p95 after",
            "Server p50",
            "p50 change",
        ],
        rows,
    )
    c["n_warm"] = str(after["warm_client_ms"]["/predict"]["n"])
    for k, d in (("before", before), ("code", code), ("after", after)):
        c[f"cold_{k}"] = f"{d['cold_start_to_ready_ms']['median'] / 1000:.1f} s"
    for k in ("container_v09", "container_d5"):
        d = R.get(k)
        c[k] = f"{d['cold_start_to_ready_ms']['median'] / 1000:.1f} s" if d else PENDING
        c[f"{k}_mem"] = d["idle_memory"][0] if d else PENDING
    st = R["container_d5"]["startup_ms_reported_by_ready"] if "container_d5" in R else {}
    c["ct_model_ms"] = f"{st.get('model', 0):.0f} ms"
    c["ct_assistant_ms"] = f"{st.get('assistant', 0) / 1000:.1f} s"
    c["ct_total_ms"] = f"{st.get('total', 0) / 1000:.1f} s"
    c["ct_warm_ms"] = f"{st.get('warm_up', 0):.0f} ms"
    loads = after["artifact_load_ms"]
    c["load_model_cold"] = f"{loads['model.joblib'] / 1000:.1f} s"
    c["load_model_warm"] = f"{loads['reload: model.joblib']:.0f} ms"
    c["load_embed_cold"] = f"{loads['embedding model (ONNX)'] / 1000:.1f} s"
    c["load_embed_warm"] = f"{loads['reload: embedding model (ONNX)'] / 1000:.1f} s"
    c["load_rules"] = f"{loads['reload: rules.json -> RuleEngine']:.0f} ms"
    p_after = after["warm_client_ms"]["/predict"]["p50"]
    c["lazy_factor"] = f"{(loads['reload: model.joblib'] + p_after) / p_after:.0f}×"
    c["lazy_chat_factor"] = (
        f"{loads['reload: embedding model (ONNX)'] / after['warm_client_ms']['/chat policy (rule)']['p50']:.0f}×"  # noqa: E501
    )
    c["predict_before"] = _ms(before["warm_client_ms"]["/predict"]["p50"])
    c["predict_code"] = _ms(code["warm_client_ms"]["/predict"]["p50"])
    c["predict_after"] = _ms(p_after)
    c["report_before"] = _ms(before["warm_client_ms"]["/monitoring/report"]["p50"])
    c["report_after"] = _ms(after["warm_client_ms"]["/monitoring/report"]["p50"])
    c["products_before"] = _ms(before["warm_client_ms"]["/products"]["p50"])
    c["products_after"] = _ms(after["warm_client_ms"]["/products"]["p50"])
    c["first_predict_before"] = _ms(before["first_call_ms"]["/predict"])
    c["first_predict_after"] = _ms(after["first_call_ms"]["/predict"])
    live = R.get("live_v09")
    if live:
        w = live["warm_client_ms"]
        c["live_table"] = _md(
            ["Endpoint (live, v0.9)", "p50 ms", "p95 ms"],
            [[f"`{k}`", _ms(v["p50"]), _ms(v["p95"])] for k, v in w.items()],
        )
        c["live_health"] = _ms(w["/health"]["p50"])
        c["live_predict"] = _ms(w["/predict"]["p50"])
    else:
        c["live_table"], c["live_health"], c["live_predict"] = PENDING, PENDING, PENDING
    c["machine"] = after["machine"]


def _scalability(c: dict[str, str], sc: dict) -> dict[str, float]:
    cells = {(x["endpoint"], x["concurrency"]): x for x in sc["throughput"]["cells"]}
    pred = max(v["rps"] for (e, _), v in cells.items() if e == "/predict")
    batch = max(v["rps"] for (e, _), v in cells.items() if e == "/predict/batch (100)")
    rec = max(v["rps"] for (e, _), v in cells.items() if e == "/recommend")
    chat = max(v["rps"] for (e, _), v in cells.items() if e == "/chat policy (rule)")
    c["tp_table"] = _md(
        ["Endpoint", "1 client", "4 clients", "16 clients", "p95 at 16"],
        [
            [
                f"`{e}`",
                *(f"{cells[(e, k)]['rps']:.0f} req/s" for k in (1, 4, 16)),
                f"{cells[(e, 16)]['p95']:.0f} ms",
            ]
            for e in ("/predict", "/predict/batch (100)", "/recommend", "/chat policy (rule)")
        ],
    )
    c["tp_predict"], c["tp_batch_rows"] = f"{pred:.0f}", f"{batch * 100:,.0f}"
    c["tp_recommend"], c["tp_chat"] = f"{rec:.0f}", f"{chat:.0f}"
    c["tp_predict_ar"] = f"{pred * CPU_DERATE:.0f}"
    c["p95_predict_16"] = f"{cells[('/predict', 16)]['p95']:.0f}"
    sco = {r["n"]: r for r in sc["scoring"]["rows"]}
    c["score_100k"] = f"{sco[100_000]['seconds']:.2f} s"
    c["score_rate"] = f"{sco[100_000]['rows_per_s']:,.0f}"
    rules = sc["rules"]["rows"]
    c["rules_table"] = _md(
        ["Rules served", "/recommend p95", "Rules checked / basket", "Memory", "Load"],
        [
            [
                f"{r['n_rules']:,} (×{r['factor']})",
                f"{r['p95']:.2f} ms",
                f"{r['rules_checked_mean']:,}",
                f"{r['memory_mb']:.0f} MB",
                f"{r['build_s']:.2f} s",
            ]
            for r in rules
        ],
    )
    c["rules_mined"] = f"{sc['rules']['mined_rules']:,}"
    r64 = rules[-1]
    c["rules_64_p95"], c["rules_64_n"] = f"{r64['p95']:.0f} ms", f"{r64['n_rules']:,}"
    c["rules_64_mem"] = f"{r64['memory_mb']:.0f} MB"
    ret = {r["n_chunks"]: r for r in sc["retrieval"]["rows"]}
    c["ret_current"] = str(sc["retrieval"]["current_chunks"])
    c["ret_100k"] = f"{ret[100_000]['ms']:.0f} ms"
    c["ret_1m"], c["ret_1m_mem"] = (
        f"{ret[1_000_000]['ms']:.0f} ms",
        f"{ret[1_000_000]['memory_mb'] / 1000:.1f} GB",
    )
    log = sc["logstore"]
    rep = {r["window"]: r for r in log["report"]}
    c["log_bytes"] = f"{log['bytes_per_prediction_and_request']:,}"
    c["rep_500"], c["rep_10k"], c["rep_50k"] = (
        f"{rep[w]['report_ms']:.0f} ms" for w in (500, 10_000, 50_000)
    )
    gb_year_1rps = log["bytes_per_prediction_and_request"] * 86_400 * 365 / 1e9
    c["log_gb_year_1rps"] = f"{gb_year_1rps:.0f} GB"
    return {"pred_rps": pred, "chat_rps": chat}


def _costs(c: dict[str, str], tp: dict[str, float], tok: dict | None, aws: dict | None) -> None:
    hour_active = VCPU * PRICES["apprunner_vcpu_h"] + GB * PRICES["apprunner_gb_h"]
    hour_idle = GB * PRICES["apprunner_gb_h"]
    c["ar_active_h"], c["ar_idle_h"] = f"${hour_active:.3f}", f"${hour_idle:.3f}"
    c["ar_idle_month"] = _money(hour_idle * HOURS_PER_MONTH, 2)
    c["ar_active_month"] = _money(hour_active * HOURS_PER_MONTH, 2)
    rps = tp["pred_rps"] * CPU_DERATE
    marginal = hour_active / (rps * 3600) * 1000
    c["cost_1k_pred_marginal"] = f"${marginal:.6f}"
    rows = []
    for n in (10_000, 1_000_000, 100_000_000):
        busy_h = n / (rps * 3600)
        monthly = hour_idle * HOURS_PER_MONTH + busy_h * VCPU * PRICES["apprunner_vcpu_h"]
        inst = max(1, int(np.ceil(n / (rps * 3600 * HOURS_PER_MONTH * 0.6))))
        monthly = max(monthly, inst * hour_active * HOURS_PER_MONTH) if inst > 1 else monthly
        rows.append([f"{n:,}", f"{inst}", _money(monthly, 2), f"${monthly / n * 1000:.4f}"])
    c["cost_pred_table"] = _md(
        [
            "Predictions / month",
            "Instances (≤ 60% busy)",
            "App Runner / month",
            "Per 1,000 predictions",
        ],
        rows,
    )
    rule_cpu_s = 1 / (tp["chat_rps"] * CPU_DERATE)
    rule_1k = 1000 * rule_cpu_s / 3600 * VCPU * PRICES["apprunner_vcpu_h"]
    ollama_ms = tok["latency_ms_p50"] if tok else 1550.0
    ollama_1k = 1000 * ollama_ms / 1000 / 3600 * PRICES["g4dn_xlarge_h"]
    openai_1k = tok["openai_est_usd_per_1000_queries"] if tok else None
    c["tok_in"] = f"{tok['mean_prompt_tokens']:.0f}" if tok else PENDING
    c["tok_out"] = f"{tok['mean_completion_tokens']:.0f}" if tok else PENDING
    c["ollama_ms"] = f"{ollama_ms / 1000:.1f} s"
    c["g4dn_month"] = _money(PRICES["g4dn_xlarge_h"] * HOURS_PER_MONTH, 0)
    c["assistant_cost_table"] = _md(
        [
            "Provider",
            "Where it runs",
            "Per 1,000 queries (marginal)",
            "Fixed cost",
            "Measured quality (D4)",
        ],
        [
            [
                "rule",
                "API container (CPU)",
                f"${rule_1k:.4f}",
                "none beyond the API",
                "tool choice 100% golden / 35% paraphrase",
            ],
            [
                "ollama (llama3.1:8b)",
                "GPU host, e.g. g4dn.xlarge",
                f"${ollama_1k:.2f}",
                f"{c['g4dn_month']}/month if always on",
                "97.5% / 95%",
            ],
            [
                "openai (gpt-4o-mini)",
                "hosted, outside Québec",
                f"${openai_1k:.2f}" if openai_1k else PENDING,
                "none",
                "not evaluated (no key)",
            ],
        ],
    )
    c["ollama_1k"], c["openai_1k"] = (
        f"${ollama_1k:.2f}",
        f"${openai_1k:.2f}" if openai_1k else PENDING,
    )
    c["aws_limit"] = "" if aws else "Total cloud spend is PENDING until Cost Explorer is queried. "
    if aws:
        c["aws_total"] = _money(aws["total"], 2)
        c["aws_period"] = f"{aws['period']['start']} to {aws['period']['end_exclusive']}"
        c["aws_table"] = _md(
            ["Service", "Spend (USD)"], [[k, f"${v:.2f}"] for k, v in aws["by_service"].items()]
        )
    else:
        c["aws_total"], c["aws_period"] = PENDING, PENDING
        c["aws_table"] = (
            "*Cost Explorer not yet queried: run `aws sso login --profile admin-profile` then "
            "`python scripts/collect_aws_costs.py`; this table is generated from its output.*"
        )


def _fairness(c: dict[str, str], fair: dict, old: dict | None) -> None:
    v, t = fair["policies"]["validation"], fair["policies"]["test"]
    rows = []
    for name in v:
        sv, st = v[name], t[name]
        rows.append(
            [
                name.replace("_", " "),
                f"{sv['pr_auc']:.3f}",
                f"{sv['coupons']:,} / {st['coupons']:,}",
                f"{sv['lapser_recall']:.2f} / {st['lapser_recall']:.2f}",
                f"{sv['top_quartile_recall']:.2f} / {st['top_quartile_recall']:.2f}",
                f"{sv['worst_gap']:.2f} / {st['worst_gap']:.2f}",
                f"{_money(sv['profit'])} / {_money(st['profit'])}",
            ]
        )
    c["mitigation_table"] = _md(
        [
            "Policy",
            "PR-AUC (val)",
            "Coupons val / test",
            "Lapser recall",
            "Top-quartile recall",
            "Worst gap",
            "Campaign profit val / test",
        ],
        rows,
    )
    audit = v["M0_single_threshold"]["audit"]
    arows = []
    for g in audit["groups"]:
        arows.append(
            [
                g["dimension"].replace("_", " "),
                g["group"],
                f"{g['n']:,}",
                f"{g['n_lapsers']:,}",
                _pct(g["selection_rate"]),
                _pct(g["lapser_recall"]),
                _pct(g["coupon_precision"]),
                "—" if g["pr_auc"] is None else f"{g['pr_auc']:.3f}",
                ("**flagged** " if g["flagged"] else "")
                + ("" if g["reliable"] else "(< 30 lapsers)")
                + ("" if g["gap"] is None else f"{-100 * g['gap']:+.0f} pts"),
            ]
        )
    o = audit["overall"]
    arows.append(
        [
            "**overall**",
            "",
            f"{o['n']:,}",
            f"{o['n_lapsers']:,}",
            _pct(o["selection_rate"]),
            _pct(o["lapser_recall"]),
            _pct(o["coupon_precision"]),
            f"{v['M0_single_threshold']['pr_auc']:.3f}",
            "",
        ]
    )
    c["audit_table"] = _md(
        [
            "Dimension",
            "Group",
            "Customers",
            "Lapsers",
            "Selection rate",
            "Lapser recall",
            "Coupon precision",
            "PR-AUC",
            "vs overall recall",
        ],
        arows,
    )
    m0v, m0t = v["M0_single_threshold"], t["M0_single_threshold"]
    c["overall_recall"], c["top_recall"] = (
        _pct(m0v["lapser_recall"]),
        _pct(m0v["top_quartile_recall"]),
    )
    c["worst_gap_val"], c["worst_gap_test"] = (
        f"{100 * m0v['worst_gap']:.0f}",
        f"{100 * m0t['worst_gap']:.0f}",
    )
    c["flagged_val"] = ", ".join(f"`{f}`" for f in m0v["flagged"])
    c["flagged_test"] = ", ".join(f"`{f}`" for f in m0t["flagged"])
    c["missed_top"] = f"{m0v['missed_top_lapsers']:,}"
    c["missed_top_value"] = _money(m0v["missed_top_value"])
    c["threshold"] = f"{fair['threshold']:.2f}"
    c["spend_edges"] = ", ".join(f"${e:,.0f}" for e in fair["spend_edges_train"])
    th = fair["group_thresholds_M2"]
    c["m2_thresholds"] = ", ".join(f"{k.split(' ')[0]} {x:.2f}" for k, x in th.items())
    if old:
        o9 = old["policies"]["validation"]["M0_single_threshold"]
        o9t = old["policies"]["test"]["M0_single_threshold"]
        c["v09_gap_val"], c["v09_gap_test"] = (
            f"{100 * o9['worst_gap']:.0f}",
            f"{100 * o9t['worst_gap']:.0f}",
        )
        c["v09_top_val"] = _pct(o9["top_quartile_recall"])
    else:
        c["v09_gap_val"] = c["v09_gap_test"] = c["v09_top_val"] = PENDING


def _model(c: dict[str, str], paths: Paths) -> None:
    m = paths.metrics
    mini = read_json(m / "minimisation.json")
    rows = [
        [
            g["group"].replace("_", " "),
            ", ".join(f"`{f}`" for f in g["features"]),
            f"{g['cv_loss']:+.4f}",
            f"{g['val_loss']:+.4f}",
            f"{g['folds_worse']}/5",
            "keep" if g["necessary"] else "**remove**",
            g["why_audited"],
        ]
        for g in mini["groups"]
    ]
    c["minimisation_table"] = _md(
        [
            "Group",
            "Features",
            "CV PR-AUC loss",
            "Validation loss",
            "Folds worse",
            "Decision",
            "Why audited",
        ],
        rows,
    )
    comb, full = mini["combined_without_removable"], mini["full"]
    c["mini_full_cv"], c["mini_comb_cv"] = f"{full['cv_pr_auc']:.4f}", f"{comb['cv_pr_auc']:.4f}"
    c["mini_full_val"], c["mini_comb_val"] = (
        f"{full['val_pr_auc']:.4f}",
        f"{comb['val_pr_auc']:.4f}",
    )
    c["n_all"], c["n_model"], c["n_removed"] = (
        str(len(FEATURES)),
        str(len(MODEL_FEATURES)),
        str(len(EXCLUDED_FROM_MODEL)),
    )
    c["model_features"] = ", ".join(f"`{f}`" for f in MODEL_FEATURES)
    dec = read_json(m / "retrain_decision.json")
    c["gate_table"] = _md(
        ["Gate", "Result"],
        [[k.replace("_", " "), "pass" if ok else "**fail**"] for k, ok in dec["checks"].items()],
    )
    c["gate_champ_pr"], c["gate_cand_pr"] = (
        f"{dec['champion']['pr_auc']:.4f}",
        f"{dec['candidate']['pr_auc']:.4f}",
    )
    meta = read_json(paths.model_meta)
    ho = meta["promotion"].get("human_override", {})
    c["registry_prev"] = str(ho.get("previous_champion_version", "—"))
    c["registry_new"] = str(meta["registry"]["version"])
    c["model_version"] = meta["model_version"]
    t3, t2 = read_json(m / "test_evaluation.json"), _opt(m / "test_evaluation_look2.json")
    c["test_pr_new"] = f"{t3['metrics']['pr_auc']:.3f}"
    c["test_ci_new"] = f"[{t3['ci95']['pr_auc']['lo']:.3f}, {t3['ci95']['pr_auc']['hi']:.3f}]"
    c["test_looks"] = str(t3["n_test_looks"])
    c["ece_new"] = f"{t3['metrics']['ece']:.3f}"
    c["threshold_new"] = f"{meta['decision_threshold']:.2f}"
    c["test_profit_new"] = _money(t3["policy"]["model_profit"])
    c["test_vs_everyone_new"] = _money(t3["policy"]["model_vs_everyone"])
    if t2:
        c["test_pr_old"] = f"{t2['metrics']['pr_auc']:.3f}"
        c["threshold_d3"] = f"{t2['policy']['threshold']:.2f}"
        c["test_profit_d3"] = _money(t2["policy"]["model_profit"])


def _interp(c: dict[str, str], it: dict) -> None:
    g = {x["feature"]: x for x in it["global"]}
    top = it["global"][:5]
    c["imp_top"] = ", ".join(f"`{x['feature']}`" for x in top)
    c["imp_top1"] = top[0]["feature"]
    c["imp_top1_share"] = _pct(top[0]["share_top1"])
    c["additivity"] = f"{it['additivity_max_abs_error']:.1e}"
    c["base_p"] = f"{it['base_p']:.2f}"
    c["season_top3"] = _pct(it["season_q_in_top3_if_not_excluded"])
    c["season_perm"] = f"{g['season_q']['perm_pr_auc_drop']:.4f}"
    st = it["stability"]
    c["stab_top3"], c["stab_top1"] = _pct(st["top3_set_identical"]), _pct(st["top1_identical"])
    c["stab_jacc"] = f"{st['top3_mean_jaccard']:.2f}"
    d = it["direction_checks"]
    c["dir_ok"] = f"{sum(v['agrees'] for v in d.values())}/{len(d)}"
    c["dir_table"] = ", ".join(f"`{k}` {v['corr_value_vs_effect']:+.2f}" for k, v in d.items())
    c["max_perm"] = f"{max(x['perm_pr_auc_drop'] for x in it['global']):.3f}"
    ex = it["examples"]
    c["examples_table"] = _md(
        ["Customer", "P(return)", "Proposed", "Outcome", "Principal factors (effect on log-odds)"],
        [
            [
                lab,
                f"{e['p_return']:.2f}",
                "coupon" if e["send_coupon"] else "no coupon",
                "returned" if e["y"] else "lapsed",
                "; ".join(
                    f"{f['label'].lower()} = {f['value']:.0f} ({f['effect']:+.2f})"
                    if isinstance(f["value"], float)
                    else f"{f['feature']} ({f['effect']:+.2f})"
                    for f in e["factors"]
                ),
            ]
            for lab, e in zip(
                ("lowest score", "at the threshold", "highest score"), ex, strict=True
            )
        ],
    )
    c["n_ever_top3"] = str(it["n_features_ever_top3"])


def d5_context(paths: Paths) -> dict[str, str]:
    m = paths.metrics
    c: dict[str, str] = {}
    lp = _opt(m / "latency_profile.json")
    if not lp or not {"before", "after", "after_code"} <= set(lp["runs"]):
        return c  # D5 evidence not collected yet: only the D5 build needs these keys
    _latency(c, lp)
    latency_figure(paths, lp["runs"]["before"], lp["runs"]["after"])
    sc = read_json(m / "scalability.json")
    tp = _scalability(c, sc)
    scalability_figure(paths, sc)
    _costs(c, tp, _opt(m / "assistant_tokens.json"), _opt(m / "aws_costs.json"))
    fair = read_json(m / "fairness.json")
    _fairness(c, fair, _opt(m / "fairness_v09_37features.json"))
    fairness_figure(paths, fair)
    _model(c, paths)
    it = read_json(m / "interpretability.json")
    _interp(c, it)
    importance_figure(paths, it)
    p = load_params()
    c["cache_s"] = str(p.monitoring.report_cache_seconds)
    c["save_rate"] = _pct(p.business.save_rate)
    c["cpu_derate"] = f"{CPU_DERATE:.1f}"
    c["price_vcpu"], c["price_gb"] = (
        f"${PRICES['apprunner_vcpu_h']}",
        f"${PRICES['apprunner_gb_h']}",
    )
    c["price_g4dn"] = f"${PRICES['g4dn_xlarge_h']}"
    c["price_openai"] = (
        f"${p.assistant.openai_price_per_mtok.input} / ${p.assistant.openai_price_per_mtok.output}"
    )
    return c
