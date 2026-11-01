"""D7 template values: the defence brief and the reflection reuse every D2-D6 key; this module adds
the readiness check (``d7_evidence.json``, scripts/collect_d7_evidence.py) and the served campaign
policy (``campaign_policy.json``, retention/campaign.py). Missing evidence renders as PENDING.
"""

from __future__ import annotations

from typing import Any

from boreal_marche.config import Paths
from boreal_marche.reporting.d6 import PENDING, _opt, money

MARK = {"pass": "✔ pass", "fail": "✘ **fail**", "pending": "… pending", "manual": "☐ manual"}


def _pct(x: float | None) -> str:
    return PENDING if x is None else f"{100 * x:.0f}%"


def _checklist(ev: dict[str, Any]) -> str:
    rows = ["| Area | Check | Status | Evidence |", "|---|---|---|---|"]
    for r in ev.get("checks", []):
        detail = r["detail"].replace("|", "/")[:90]
        rows.append(f"| {r['group']} | {r['item']} | {MARK[r['status']]} | {detail} |")
    return "\n".join(rows)


def _campaign(c: dict[str, str], camp: dict[str, Any] | None) -> None:
    keys = ("review", "review_lapsers", "q4_auto", "q4_review", "q3_auto", "q3_review", "hold",
            "hold_share", "mde", "quarters", "forgone", "proposed_profit", "review_profit",
            "p_hold", "p_treat", "model_recall", "rule_recall", "rule_coupons", "rule_profit",
            "proposed")  # fmt: skip
    if camp is None:
        c.update({f"camp_{k}_{s}": PENDING for k in keys for s in ("val", "test")})
        c["camp_q3_th"] = c["camp_q4_th"] = c["camp_holdout_share"] = PENDING
        return
    pol = camp["policy"]
    c["camp_q3_th"] = f"{pol['review_thresholds']['Q3']:.2f}"
    c["camp_q4_th"] = f"{pol['review_thresholds']['Q4 (top)']:.2f}"
    c["camp_holdout_share"] = f"{100 * pol['holdout_share']:.0f}%"
    c["camp_id"] = pol["campaign_id"]
    for snap, s in (("val", camp["snapshots"]["validation"]), ("test", camp["snapshots"]["test"])):
        q = s["by_spend_quartile"]
        h = s["holdout"]
        c |= {
            f"camp_review_{snap}": f"{s['review_list']['n']:,}",
            f"camp_review_lapsers_{snap}": _pct(s["review_list"]["lapser_share"]),
            f"camp_q4_auto_{snap}": f"{q['Q4 (top)']['lapser_recall_automatic']:.2f}",
            f"camp_q4_review_{snap}": f"{q['Q4 (top)']['lapser_recall_if_review_approved']:.2f}",
            f"camp_q3_auto_{snap}": f"{q['Q3']['lapser_recall_automatic']:.2f}",
            f"camp_q3_review_{snap}": f"{q['Q3']['lapser_recall_if_review_approved']:.2f}",
            f"camp_hold_{snap}": f"{h['n']:,}",
            f"camp_hold_share_{snap}": _pct(h["share_of_proposed"]),
            f"camp_mde_{snap}": f"{h['min_detectable_save_rate_one_quarter']:.2f}",
            f"camp_quarters_{snap}": str(h["quarters_to_detect_assumed_save_rate"]),
            f"camp_forgone_{snap}": money(h["profit_forgone"]),
            f"camp_proposed_profit_{snap}": money(h["profit_proposed"]),
            f"camp_review_profit_{snap}": money(s["review_list"]["profit_if_all_approved"]),
            f"camp_p_hold_{snap}": f"{h['mean_p_holdout']:.2f}",
            f"camp_p_treat_{snap}": f"{h['mean_p_treated']:.2f}",
            f"camp_model_recall_{snap}": f"{s['lapser_recall_model']:.2f}",
            f"camp_rule_recall_{snap}": f"{s['rule_recency60']['lapser_recall']:.2f}",
            f"camp_rule_coupons_{snap}": f"{s['rule_recency60']['coupons']:,}",
            f"camp_rule_profit_{snap}": money(s["rule_recency60"]["profit"]),
            f"camp_proposed_{snap}": f"{s['actions']['coupon'] + s['actions']['holdout']:,}",
        }


def d7_context(paths: Paths, base: dict[str, str]) -> dict[str, str]:
    m = paths.metrics
    ev = _opt(m / "d7_evidence.json") or {}
    c: dict[str, str] = {}
    _campaign(c, _opt(m / "campaign_policy.json"))
    live = ev.get("live", {})
    v = live.get("version") or {}
    c["d7_checked"] = ev.get("collected_at", PENDING)[:16].replace("T", " ") + " UTC"
    c["d7_api_version"] = v.get("api_version", PENDING)
    c["d7_commit"] = (v.get("git_commit") or PENDING)[:7]
    c["d7_master"] = (live.get("master_head") or PENDING)[:7]
    c["d7_checklist"] = _checklist(ev)
    video = ev.get("defence_video") or {}
    c["d7_defence_video"] = video.get("url", PENDING)
    c["d7_defence_video_title"] = video.get("title") or PENDING
    st = [r["status"] for r in ev.get("checks", [])]
    c["d7_n_pass"], c["d7_n_fail"] = str(st.count("pass")), str(st.count("fail"))
    c["d7_n_pending"], c["d7_n_manual"] = str(st.count("pending")), str(st.count("manual"))
    t = ev.get("tests") or (_opt(m / "d6_evidence.json") or {}).get("tests") or {}
    c["d7_tests"] = str(t.get("passed", PENDING))
    c["d7_cov"] = f"{t['coverage_pct']}%" if t else PENDING
    tags = (ev.get("repository") or {}).get("tags") or []
    c["d7_tags"] = ", ".join(f"`{x}`" for x in tags) or PENDING
    q = ev.get("sagemaker_quota") or {}
    c["d7_quota"] = f"{q['Status']}, case {q['CaseId']}" if q else PENDING
    c["d7_quota_approved"] = "yes" if q.get("Status") == "APPROVED" else "no"
    hpo = _opt(m / "cloud_hpo_evidence_aws.json") or {}
    c["d7_sm_result"] = (
        f"best CV PR-AUC {hpo['best_cv_pr_auc']:.4f} "
        f"against local Optuna {base['tune_xgboost_best']}"
        if hpo.get("best_cv_pr_auc") is not None
        else "not run: the quota request is still open, so every model was tuned locally"
    )
    return c
