"""Promotion check for the weekly retraining workflow (``.github/workflows/retrain.yml``).

Compares the freshly trained ``artifacts/model.joblib`` (candidate) with the committed champion
(``--champion``, extracted from ``origin/master`` by the workflow) on the **validation** snapshot —
never the test snapshot — and applies the written promotion policy
(``docs/MLFLOW_PROMOTION_POLICY.md``):

1. the candidate's own gates from ``train_final.json`` (beats the best baseline by 0.05 PR-AUC,
   calibration not worse than raw, ECE <= 0.05);
2. beats the champion by >= 0.005 validation PR-AUC (a retrain that is merely as good does not churn
   the served model);
3. **fairness gate** — subgroup recall of the coupon decision for true lapsers, by region, tenure
   band and spend quartile (groups with >= 30 lapsers): the candidate's worst gap below overall
   recall must not exceed 5 points or, if the champion already had a larger gap, must not be worse
   than the champion's (non-regression while the D5 mitigations are pending).

Writes ``reports/metrics/retrain_decision.json`` and a Markdown summary for the pull request, and
prints ``promote=true|false`` for ``$GITHUB_OUTPUT``. It never deploys: a human merges the PR.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from boreal_marche.config import Paths
from boreal_marche.features import FEATURES
from boreal_marche.retention.cv import load_gold
from boreal_marche.utils import read_json, utc_now, write_json

MIN_LAPSERS = 30
GAP_LIMIT = 0.05
MIN_GAIN = 0.005


def subgroups(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "region": df["region"].astype(str),
        "tenure_band": pd.cut(
            df["tenure_days"], [-1, 180, 365, np.inf], labels=["<6m", "6-12m", ">12m"]
        ).astype(str),
        "spend_quartile": pd.qcut(
            df["revenue_180d"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"]
        ).astype(str),
    }


def fairness(df: pd.DataFrame, send: np.ndarray) -> dict[str, Any]:
    """Recall of the coupon decision among true lapsers (y = 0), overall and per subgroup."""
    lapser = df["y"].to_numpy() == 0
    overall = float(send[lapser].mean())
    groups = []
    for dim, labels in subgroups(df).items():
        for g in sorted(labels.unique()):
            m = (labels == g).to_numpy() & lapser
            if m.sum() >= MIN_LAPSERS:
                rec = float(send[m].mean())
                groups.append({"dimension": dim, "group": g, "n_lapsers": int(m.sum()),
                               "recall": rec, "gap": overall - rec})  # fmt: skip
    worst = max((g["gap"] for g in groups), default=0.0)
    return {
        "overall_recall": overall,
        "worst_gap": worst,
        "flagged": [g for g in groups if g["gap"] > GAP_LIMIT],
        "groups": groups,
    }


def evaluate(model_path: Path, val: pd.DataFrame) -> dict[str, Any]:
    model = joblib.load(model_path)
    p = model.predict_proba(val[FEATURES])[:, 1]
    return {
        "version": model.version,
        "pr_auc": float(average_precision_score(val["y"], p)),
        "fairness": fairness(val, p <= model.threshold),
    }


def decide(cand: dict[str, Any], champ: dict[str, Any], own_gates: dict[str, bool]) -> dict:
    cw, hw = cand["fairness"]["worst_gap"], champ["fairness"]["worst_gap"]
    checks = {
        **own_gates,
        f"beats_champion_by_{MIN_GAIN}_pr_auc": cand["pr_auc"] >= champ["pr_auc"] + MIN_GAIN,
        "fairness_gap_within_limit_or_not_worse": cw <= max(GAP_LIMIT, hw) + 1e-9,
    }
    return {"checks": checks, "promote": all(checks.values())}


def summary_md(d: dict[str, Any]) -> str:
    c, h = d["candidate"], d["champion"]
    rows = "\n".join(f"| {k} | {'pass' if v else '**fail**'} |" for k, v in d["checks"].items())
    return (
        f"## Weekly retraining — {'PROMOTE (human review required)' if d['promote'] else 'no promotion'}\n\n"  # noqa: E501
        f"| | champion `{h['version']}` | candidate `{c['version']}` |\n|---|---|---|\n"
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
    cand = evaluate(Path(a.candidate) if a.candidate else paths.model_file, val)
    champ = evaluate(Path(a.champion), val)
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
