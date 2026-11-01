"""D6 template values and figures: the consolidated report, the bilingual executive summary and the
deck reuse every D2-D5 key; this module adds the live-system evidence, French-formatted amounts,
the SageMaker section and two figures.

Sources: ``d6_evidence.json`` (scripts/collect_d6_evidence.py), ``train_final.json`` and
``test_evaluation.json`` (campaign results), ``aws_costs.json`` and, once the quota is granted and
the job has run, ``cloud_hpo_evidence_aws.json`` (docs/runbooks/SAGEMAKER_HPO.md). Missing evidence
renders as PENDING.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from boreal_marche.config import Paths, load_params
from boreal_marche.features import FEATURES, MODEL_FEATURES
from boreal_marche.reporting import style
from boreal_marche.reporting.figures import _arrow, _box
from boreal_marche.utils import read_json

PENDING = "**PENDING**"
NNBSP = " "  # French thousands separator (narrow no-break space)


def _opt(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def money(x: float) -> str:
    return f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def fr_money(x: float) -> str:
    """Québec French: 216 549 $ and −1 039 $."""
    s = f"{abs(x):,.0f}".replace(",", NNBSP) + f"{NNBSP}$"
    return f"−{s}" if x < 0 else s


def k_money(x: float) -> str:
    return f"−${abs(x) / 1000:,.0f}k" if x < 0 else f"${x / 1000:,.0f}k"


def fr_k_money(x: float) -> str:
    s = f"{abs(x) / 1000:,.0f}{NNBSP}k$"
    return f"−{s}" if x < 0 else s


# ------------------------------------------------------------------------------- figures ---


def architecture_as_built(paths: Paths, c: dict[str, str]) -> None:
    """The deployed system (D2's figure showed the design, with later parts in grey)."""
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.grid(False)
    bands = [
        (4.75, 2.5, "#f3f7fd", style.BLUE,
         "OFFLINE — batch pipeline (Makefile locally; weekly in GitHub Actions)"),
        (2.55, 2.0, "#eefaf5", style.AQUA, "DELIVERY — every merge to master, GitHub Actions"),
        (0.1, 2.25, "#fdf5f1", style.ORANGE, "ONLINE — every request, milliseconds"),
    ]  # fmt: skip
    for y, h, fc, ec, title in bands:
        ax.add_patch(
            FancyBboxPatch(
                (0.1, y), 12.8, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1, ls="--"
            )
        )
        ax.text(0.25, y + h - 0.28, title, fontsize=9, weight="bold", color=ec)
    # offline
    y1, y2 = 5.95, 4.95
    _box(ax, 0.3, y1, 1.7, 0.85, "UCI Online Retail II\nSHA-256, DVC\n(CC BY 4.0)")
    _box(ax, 2.3, y1, 1.9, 0.85, "bronze → silver → gold\n11 cleaning rules\nDQ suite")
    _box(
        ax, 4.5, y1, 2.2, 0.85,
        "Head A: spot-check 10+3\nOptuna (local) · SageMaker\n"
        f"{c['n_model']} of {c['n_all']} features",
    )  # fmt: skip
    _box(
        ax, 7.0, y1, 2.3, 0.85,
        f"calibrate (sigmoid)\nprofit threshold {c['threshold']}\nMLflow @champion",
    )  # fmt: skip
    _box(ax, 4.5, y2, 2.2, 0.75, f"Head B: FP-Growth\n{c['b_rules']} rules, lift ≥ {c['b_lift']}")
    _box(
        ax, 0.3, y2, 1.7, 0.75,
        f"Head C corpus\n{c['corpus_docs']} docs → {c['n_chunks']} chunks\n→ index/",
    )  # fmt: skip
    _box(
        ax, 9.7, y2 + 0.05, 3.0, 1.7,
        "artifacts (in the image)\nmodel.joblib + model_meta.json\nrules.json · index/\n"
        "reference_profile.json",
        fc="#ffffff", ec=style.AQUA, size=7.2,
    )  # fmt: skip
    for a, b in [
        ((2.0, 6.37), (2.3, 6.37)),
        ((4.2, 6.37), (4.5, 6.37)),
        ((6.7, 6.37), (7.0, 6.37)),
    ]:
        _arrow(ax, a, b)
    _arrow(ax, (3.25, y1), (4.5, 5.32))
    _arrow(ax, (9.3, 6.37), (9.7, 6.37), color=style.AQUA)
    _arrow(ax, (6.7, 5.32), (9.7, 5.32), color=style.AQUA)
    # delivery
    yd = 2.85
    _box(ax, 0.3, yd, 1.6, 0.95, "pull request\n(branch rule:\nno direct push)", ec=style.AQUA)
    _box(
        ax, 2.2, yd, 2.5, 0.95,
        f"CI: ruff · mypy strict\n{c['d6_tests']} tests, coverage gate\nimage build · gitleaks",
        ec=style.AQUA,
    )  # fmt: skip
    _box(
        ax,
        5.0,
        yd,
        2.1,
        0.95,
        "CD: OIDC role\n(no AWS key stored)\n1-hour credentials",
        ec=style.AQUA,
    )
    _box(ax, 7.4, yd, 1.9, 0.95, "ECR (us-east-1)\nimage tag =\ncommit SHA", ec=style.AQUA)
    _box(
        ax, 9.6, yd, 3.1, 0.95,
        "smoke test: /version = merge,\ngolden prediction, rule fires,\nassistant cites a source",
        ec=style.AQUA,
    )  # fmt: skip
    for a, b in [
        ((1.9, 3.32), (2.2, 3.32)),
        ((4.7, 3.32), (5.0, 3.32)),
        ((7.1, 3.32), (7.4, 3.32)),
    ]:
        _arrow(ax, a, b)
    _arrow(ax, (9.3, 3.32), (9.6, 3.32))
    _arrow(ax, (10.3, 5.0), (8.6, 3.8), "baked into the image", color=style.AQUA)
    # online
    _box(
        ax, 0.3, 0.4, 2.3, 1.15,
        "Streamlit Community Cloud\nEN/FR · 5 sections\n(WebSocket — not App Runner)",
        ec=style.ORANGE,
    )  # fmt: skip
    _box(
        ax, 3.0, 0.3, 4.6, 1.4,
        "AWS App Runner (us-east-1) — FastAPI, 1 vCPU / 2 GB\n"
        "A /predict(?explain)  ·  B /recommend  ·  C /chat (rule provider)\n"
        "/health /ready /version · /metrics · /monitoring/*",
        ec=style.ORANGE, size=7.2,
    )  # fmt: skip
    _box(
        ax, 8.0, 0.4, 2.2, 1.15,
        "prediction log →\nPSI/KS drift, quality,\nlatency, alerts",
        ec=style.ORANGE,
    )  # fmt: skip
    _box(
        ax, 10.55, 0.4, 2.15, 1.15,
        "marketing reviewer\napproves every list\n(Law 25 s. 12.1)",
        ec=style.ORANGE,
    )  # fmt: skip
    _arrow(ax, (2.6, 0.98), (3.0, 0.98), "HTTPS")
    _arrow(ax, (7.6, 0.98), (8.0, 0.98))
    _arrow(ax, (10.2, 0.98), (10.55, 0.98))
    _arrow(ax, (8.35, yd), (6.2, 1.7), "deploy", color=style.AQUA)
    ax.text(
        12.85, 0.15,
        "SageMaker HPO: spot quota pending (ca-central-1) · Ollama: local / Compose only",
        fontsize=6.5, ha="right", color=style.INK_2,
    )  # fmt: skip
    fig.tight_layout()
    fig.savefig(paths.figures / "d6_architecture.png", dpi=170)
    plt.close(fig)


def campaign_figure(paths: Paths, val: dict, test: dict) -> None:
    """Quarterly campaign result per policy: the business picture of Head A."""
    pols = [
        ("Coupon to everyone\n(today)", "everyone_profit", style.NEUTRAL),
        ("Model list\n(threshold)", "model_profit", style.BLUE),
        ("Nobody", None, style.GRID),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for ax, (title, d) in zip(
        axes,
        (("Validation quarter (Jun 2011)", val), ("Test quarter (Sep 2011)", test)),
        strict=True,
    ):
        vals = [d[k] if k else 0.0 for _, k, _ in pols]
        x = np.arange(len(pols))
        ax.bar(x, vals, 0.6, color=[col for *_, col in pols], edgecolor=style.INK_2, lw=0.4)
        for xi, v in zip(x, vals, strict=True):
            ax.text(
                xi, v + (1500 if v >= 0 else -1500), money(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7.5, color=style.INK,
            )  # fmt: skip
        ax.axhline(0, color=style.INK_2, lw=0.8)
        ax.set_xticks(x, [p for p, *_ in pols], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("campaign profit ($)")
    axes[0].yaxis.set_major_formatter(lambda v, _: k_money(v))
    lo = min(val["everyone_profit"], test["everyone_profit"])
    axes[0].set_ylim(lo * 1.3, max(val["model_profit"], 1) + 12_000)
    fig.suptitle("One quarterly coupon campaign, by policy", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(paths.figures / "d6_campaign.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------------------- context ---


def _sagemaker(c: dict[str, str], paths: Paths, tune: dict, base: dict[str, str]) -> None:
    ev = _opt(paths.metrics / "cloud_hpo_evidence_aws.json")
    loc = tune["results"]["xgboost"]
    if ev and ev.get("best_cv_pr_auc"):
        c["sm_status"] = f"run {ev.get('date', '')}: best CV PR-AUC {ev['best_cv_pr_auc']:.4f}"
        c["sm_section"] = "\n".join(
            [
                f"The managed job `{ev.get('tuning_job_name', '')}` ran on {ev.get('date', '')} "
                "(Bayesian, managed spot, `ml.m5.large`, `ca-central-1`) on the same "
                f"{len(MODEL_FEATURES)} features and folds as the local search.",
                "",
                "| | Local Optuna | SageMaker HPO |",
                "|---|---:|---:|",
                f"| Best CV PR-AUC | {loc['best_value']:.4f} | {ev['best_cv_pr_auc']:.4f} |",
                f"| Wall-clock | {base['tune_wall_xgboost']} min | "
                f"{ev.get('wall_clock_minutes', 0):.1f} min |",
                f"| Billable seconds | — | {ev.get('billable_seconds_total', 0):,} "
                f"(spot savings {ev.get('spot_savings_percent', 0):.0f}%) |",
                f"| Cost | $0 | ${ev.get('cost_usd_from_billing', 0):.2f} |",
            ]
        )
    else:
        c["sm_status"] = "spot quota pending (case 179073448000753, opened 2026-09-29)"
        c["sm_section"] = (
            "The Service Quotas request for `ml.m5.large` spot training in `ca-central-1` (case "
            "179073448000753, opened 2026-09-29) was still open when this report was built, so "
            "**no managed job has run** and none is reported. All tuning used the local Optuna "
            f"backend ({base['trials_per_model']} trials per model, "
            f"{base['tune_wall_xgboost']} min "
            "for XGBoost, $0). The SageMaker path is implemented and tested short of AWS: the "
            "script-mode entry point (`retention/sm_train.py`) reproduces the local CV score, the "
            "self-contained source directory runs with `python -I`, and "
            "`python scripts/run.py hpo --provider aws --dry-run` prints the job (Bayesian, 20 "
            "jobs, 2 parallel, managed spot, `max_run` 15 min, `max_wait` 30 min). When the "
            "quota is granted, the run writes `reports/metrics/cloud_hpo_evidence_aws.json` and "
            "this section is regenerated with its numbers."
        )


def d6_context(paths: Paths, base: dict[str, str]) -> dict[str, str]:
    m = paths.metrics
    c: dict[str, str] = {}
    ev = _opt(m / "d6_evidence.json") or {}
    live = ev.get("live") or {}
    ver = live.get("version") or {}
    c["d6_ui"] = live.get("ui_url") or PENDING
    c["d6_api"] = live.get("api_url") or PENDING
    c["d6_docs"] = f"{live['api_url']}/docs" if live.get("api_url") else PENDING
    c["d6_commit"] = ver.get("git_commit", "")[:7] or PENDING
    c["d6_api_version"] = ver.get("api_version", PENDING)
    c["d6_model_version"] = ver.get("model_version", PENDING)
    c["d6_checked"] = live.get("checked_at", PENDING)[:10]
    c["d6_ready"] = str(live.get("ready_status", PENDING))
    c["d6_talk"] = (ev.get("presentation_video") or {}).get("url") or PENDING
    t = ev.get("tests", {})
    c["d6_tests"] = str(t.get("passed", base.get("tests_passed", "—")))
    c["d6_cov"] = f"{t.get('coverage_pct', '—')}%"
    c["d6_tags"] = ", ".join(f"`{x}`" for x in ev.get("tags", []))

    fin, test = read_json(m / "train_final.json"), read_json(m / "test_evaluation.json")
    vp, tp = fin["validation_policy"], test["policy"]
    q_val, q_test = vp["model_vs_everyone"], tp["model_vs_everyone"]
    c["annual_low"], c["annual_high"] = (
        k_money(4 * min(q_val, q_test)),
        k_money(4 * max(q_val, q_test)),
    )
    c["fr_annual_low"] = fr_k_money(4 * min(q_val, q_test))
    c["fr_annual_high"] = fr_k_money(4 * max(q_val, q_test))
    c["k_test_everyone"], c["fr_k_test_everyone"] = (
        k_money(tp["everyone_profit"]),
        fr_k_money(tp["everyone_profit"]),
    )
    c["k_val_everyone"], c["fr_k_val_everyone"] = (
        k_money(vp["everyone_profit"]),
        fr_k_money(vp["everyone_profit"]),
    )
    c["k_test_gain"], c["fr_k_test_gain"] = k_money(q_test), fr_k_money(q_test)
    c["k_val_gain"], c["fr_k_val_gain"] = k_money(q_val), fr_k_money(q_val)
    c["fr_test_profit"] = fr_money(tp["model_profit"])
    c["fr_test_everyone"], c["fr_val_everyone"] = (
        fr_money(tp["everyone_profit"]),
        fr_money(vp["everyone_profit"]),
    )
    c["fr_test_gain"], c["fr_val_gain"] = fr_money(q_test), fr_money(q_val)
    c["fr_val_profit"] = fr_money(vp["model_profit"])
    c["fr_missed_top_value"] = fr_money(read_json(m / "fairness.json")["policies"]["validation"][
        "M0_single_threshold"]["missed_top_value"])  # fmt: skip
    c["test_dummy"] = f"{test['baselines']['baseline_dummy_prior']['pr_auc']:.3f}"
    spot = read_json(m / "spotcheck.json")["table"]
    learned = [r["pr_auc_mean"] for r in spot if not r["model"].startswith("baseline_")]
    bases = [r["pr_auc_mean"] for r in spot if r["model"].startswith("baseline_")]
    c["spot_margin"] = f"{min(learned) - max(bases):.2f}"
    imgs = (_opt(m / "d4_evidence.json") or {}).get("images", {})
    c["img_api_mb"] = f"{imgs.get('boreal-api:d4', {}).get('unpacked_mb', 0):,} MB"
    c["img_v03_mb"] = f"{imgs.get('boreal-api:v0.3', {}).get('unpacked_mb', 0):,} MB"
    pol = read_json(m / "fairness.json")["policies"]["validation"]
    m0, m2 = pol["M0_single_threshold"]["coupons"], pol["M2_group_thresholds"]["coupons"]
    c["m2_extra_pct"] = f"{100 * (m2 / m0 - 1):.0f}%"
    c["test_not_sent_pct"] = f"{100 * (1 - tp['model_contacted'] / test['n']):.0f}%"
    c["val_not_sent_count"] = f"{vp['n_customers'] - vp['model_contacted']:,}"
    aws = _opt(m / "aws_costs.json")
    c["fr_aws_total"] = f"{aws['total']:.2f}".replace(".", ",") + f"{NNBSP}$" if aws else PENDING
    c["fr_ar_idle_month"] = (
        base.get("ar_idle_month", "$10").replace("$", "").replace(".", ",") + f"{NNBSP}$"
    )
    c["n_features_collected_fr"] = str(len(FEATURES))
    p = load_params()
    c["fr_save_rate"] = f"{100 * p.business.save_rate:.0f}{NNBSP}%"
    c["fr_discount"] = f"{100 * p.business.discount_rate:.0f}{NNBSP}%"
    c["fr_hold_ratio"] = base["hold_ratio"].replace(".", ",")
    for k in ("top_recall", "overall_recall", "val_not_sent_pct", "test_not_sent_pct"):
        v = c.get(k) or base.get(k, "")
        c[f"fr_{k}"] = v.replace("%", f"{NNBSP}%")
    _sagemaker(c, paths, read_json(m / "tuning.json"), base)
    campaign_figure(paths, vp, tp)
    architecture_as_built(paths, base | c)
    return c
