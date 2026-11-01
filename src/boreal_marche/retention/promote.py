"""Promotion check for the weekly retraining workflow (``.github/workflows/retrain.yml``).

Compares the freshly trained ``artifacts/model.joblib`` (candidate) with the committed champion
(``--champion``, extracted from ``origin/master`` by the workflow) on the **validation** snapshot —
never the test snapshot — and applies the written promotion policy
(``docs/MLFLOW_PROMOTION_POLICY.md``):

1. the candidate's own gates from ``train_final.json`` (beats the best baseline by 0.05 PR-AUC,
   calibration not worse than raw, ECE <= 0.05);
2. beats the champion by >= 0.005 validation PR-AUC (a retrain that is merely as good does not churn
   the served model) — or, data-minimisation clause (D5), uses strictly fewer input features and
   loses at most 0.005;
3. **fairness gate** — subgroup recall of the coupon decision for true lapsers, by region, tenure
   band and spend quartile (``retention/fairness.py``: groups with >= 30 lapsers, spend edges from
   the training snapshots): the candidate's worst gap below overall recall must not exceed 5 points
   or, if the champion already had a larger gap, must not be worse than the champion's.

Writes ``reports/metrics/retrain_decision.json`` and a Markdown summary for the pull request, and
prints ``promote=true|false`` for ``$GITHUB_OUTPUT``. It never deploys: a human merges the PR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score

from boreal_marche.config import Paths
from boreal_marche.retention.cv import load_gold
from boreal_marche.retention.fairness import GAP_LIMIT, audit, spend_edges
from boreal_marche.utils import read_json, utc_now, write_json

MIN_GAIN = 0.005


def evaluate(model_path: Path, val: pd.DataFrame, edges: list[float]) -> dict[str, Any]:
    model = joblib.load(model_path)
    p = model.predict_proba(val[model.feature_names])[:, 1]
    a = audit(val, p, p <= model.threshold, edges)
    return {
        "version": model.version,
        "n_features": len(model.feature_names),
        "pr_auc": float(average_precision_score(val["y"], p)),
        "fairness": {
            "overall_recall": a["overall"]["lapser_recall"],
            "worst_gap": a["worst_gap"],
            "flagged": a["flagged"],
            "groups": a["groups"],
        },  # fmt: skip
    }


def decide(cand: dict[str, Any], champ: dict[str, Any], own_gates: dict[str, bool]) -> dict:
    cw, hw = cand["fairness"]["worst_gap"], champ["fairness"]["worst_gap"]
    beats = cand["pr_auc"] >= champ["pr_auc"] + MIN_GAIN
    minimises = (cand["n_features"] < champ["n_features"]
                 and cand["pr_auc"] >= champ["pr_auc"] - MIN_GAIN)  # fmt: skip
    checks = {
        **own_gates,
        f"beats_champion_by_{MIN_GAIN}_pr_auc_or_minimises": beats or minimises,
        "fairness_gap_within_limit_or_not_worse": cw <= max(GAP_LIMIT, hw) + 1e-9,
    }
    return {"checks": checks, "promote": all(checks.values())}


def summary_md(d: dict[str, Any]) -> str:
    c, h = d["candidate"], d["champion"]
    rows = "\n".join(f"| {k} | {'pass' if v else '**fail**'} |" for k, v in d["checks"].items())
    return (
        f"## Weekly retraining — {'PROMOTE (human review required)' if d['promote'] else 'no promotion'}\n\n"  # noqa: E501
        f"| | champion `{h['version']}` | candidate `{c['version']}` |\n|---|---|---|\n"
        f"| input features | {h['n_features']} | {c['n_features']} |\n"
        f"| validation PR-AUC | {h['pr_auc']:.4f} | {c['pr_auc']:.4f} |\n"
        f"| lapser recall (overall) | {h['fairness']['overall_recall']:.3f} | "
        f"{c['fairness']['overall_recall']:.3f} |\n"
        f"| worst subgroup recall gap | {h['fairness']['worst_gap']:.3f} | "
        f"{c['fairness']['worst_gap']:.3f} |\n\n| gate | result |\n|---|---|\n{rows}\n\n"
        "Validation snapshot only; the test snapshot is not touched. This workflow never deploys: "
        "merging this pull request is the human approval step (Law 25 design consequence).\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--champion", required=True, help="path to the committed champion joblib")
    ap.add_argument("--candidate", default=None)
    a = ap.parse_args()
    paths = Paths()
    gold = load_gold(paths)
    val = gold[gold["split"] == "validation"].reset_index(drop=True)
    edges = spend_edges(gold[gold["split"] == "train"])
    cand = evaluate(Path(a.candidate) if a.candidate else paths.model_file, val, edges)
    champ = evaluate(Path(a.champion), val, edges)
    final = read_json(paths.metrics / "train_final.json")
    own = final["candidates"][final["champion"]]["promotion"]["checks"]
    own = {k: v for k, v in own.items() if not k.startswith("beats_current_champion")}
    decision = {"generated_at": utc_now(), "candidate": cand, "champion": champ}
    decision |= decide(cand, champ, own)
    write_json(paths.metrics / "retrain_decision.json", decision)
    (paths.reports / "retrain_summary.md").write_text(summary_md(decision), encoding="utf-8")
    print(summary_md(decision))
    print(f"promote={'true' if decision['promote'] else 'false'}")


if __name__ == "__main__":
    main()
