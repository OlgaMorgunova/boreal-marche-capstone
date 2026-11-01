# Commit and tag plan — `v0.9.5` (D5 Optimization and Ethics Report)

Same rules as [COMMIT_PLAN_TAGS.md](COMMIT_PLAN_TAGS.md) and [COMMIT_PLAN_D4.md](COMMIT_PLAN_D4.md):
branch, review every file, merge with a **merge commit**, then an **annotated** tag on `master`.
Nothing below has been run.

**Which tag.** The course (§7.1) requires tags only for D1–D4 and D7 (`v0.1`, `v0.2`, `v0.3`,
`v0.9`, `v1.0`); D5 has none. `v0.9.5` keeps the one-tag-per-deliverable history, matches the
package version (`pyproject.toml`, `__version__`, model `0.9.5+xgboost`) and sorts between `v0.9`
and `v1.0`.

**What merging does.** CD deploys the new image to App Runner: the live service will serve the
**19-feature model** and the new `/predict?explain=N`, `/ready` startup timings and cached
`/monitoring/report`. Streamlit Community Cloud redeploys the interface from the same merge, usually
a few minutes *before* App Runner finishes; for those minutes the customer list fails, because the
new interface reads `orders_180d` and the old API still returns `orders_lifetime`. It recovers by
itself when the CD run turns green.

**Not committed (ever):** `doc/` (instructor's material), `.claude/`, `.env`, `boreal.db*`,
`eval_chat.db`, `mlflow.db`, `mlruns/`, `.cache/`, `reports/*.html`.
**Delete, don't commit:** the untracked file named `: walkthrough notebook, budget-alarm evidence,
regenerated reports"` in the repository root. It is a `git diff --stat` output that a shell command
on 2026-10-25 wrote to a file by mistake.

## 0. Before committing (5 minutes)

```bash
git switch master && git pull --ff-only
git switch -c feat/d5
python scripts/run.py test lint typecheck    # expected: 89 passed, ruff clean, mypy clean
```

1. **LégisQuébec check (required by the rubric, "primary source consulted").** The site refused my
   automated read. Open https://www.legisquebec.gouv.qc.ca/en/document/cs/P-39.1, check ss. 3.1,
   3.3, 8.1, 12.1, 17, 23 and the 30-day delay in s. 32 against `GOVERNANCE_LOI25.md` §1 and §6,
   and fill in the date at `Olga, date: ____` in its header.
2. Read `docs/runbooks/SAGEMAKER_HPO.md` (your own edits, still uncommitted from D4): it contains
   your AWS account ID and role ARN. Keep them or replace them with `<account>`.

## 1. Commit 1 — your D4 documentation edits (made after `v0.9`, not by D5)

```bash
git add docs/COMMIT_PLAN_D4.md docs/COMMIT_PLAN_TAGS.md docs/RUN_FROM_ZERO_WINDOWS.md \
  docs/runbooks/AWS_APPRUNNER_DEPLOY.md docs/runbooks/SAGEMAKER_HPO.md
git diff --cached --stat
git commit -m "Docs: D4 commit plan, Windows guide and AWS runbook updates"
```

## 2. Commit 2 — D5 code, model and evidence

Latency optimisations, exact explanations, data minimisation (19-feature model retrained,
re-evaluated, reference profile and retrieval index rebuilt), fairness audit, profiling scripts,
tests.

```bash
git add \
  params.yaml pyproject.toml Makefile scripts/run.py \
  src/boreal_marche/__init__.py src/boreal_marche/api.py src/boreal_marche/config.py \
  src/boreal_marche/features.py src/boreal_marche/schemas.py \
  src/boreal_marche/monitoring/drift.py src/boreal_marche/monitoring/reference.py \
  src/boreal_marche/retention/cv.py src/boreal_marche/retention/models.py \
  src/boreal_marche/retention/promote.py src/boreal_marche/retention/sm_train.py \
  src/boreal_marche/retention/spotcheck.py src/boreal_marche/retention/train.py \
  src/boreal_marche/retention/explain.py src/boreal_marche/retention/fairness.py \
  src/boreal_marche/retention/minimise.py \
  frontend/app.py frontend/i18n.py loadtest/locustfile.py scripts/simulate_traffic.py \
  scripts/latency_profile.py scripts/scalability_probe.py scripts/assistant_token_probe.py \
  scripts/collect_aws_costs.py \
  tests/test_api.py tests/test_d5.py tests/golden/golden_predictions.json \
  artifacts/model.joblib artifacts/model_meta.json artifacts/reference_profile.json \
  artifacts/demo_customers.json artifacts/catalogue.json artifacts/index/ \
  docs/corpus/privacy.md docs/corpus/loyalty.md \
  reports/metrics/train_final.json reports/metrics/test_evaluation.json \
  reports/metrics/test_evaluation_look2.json reports/metrics/retrain_decision.json \
  reports/metrics/latency_profile.json reports/metrics/scalability.json \
  reports/metrics/interpretability.json reports/metrics/minimisation.json \
  reports/metrics/fairness.json reports/metrics/fairness_v09_37features.json \
  reports/metrics/assistant_tokens.json \
  reports/figures/profit_curve_validation.png reports/figures/reliability_validation.png \
  reports/figures/reliability_test.png
git diff --cached --stat
git commit -m "D5: latency optimisations, exact explanations, 19-feature model, fairness audit" \
  -m "Column-wise model input, folded catalogue, cached monitoring report, warm-up and startup
timings in /ready (/predict p50 7.8 -> 4.8 ms). Exact TreeSHAP factors in calibrated log-odds via
POST /predict?explain=N. Data-minimisation ablation removes 18 of 37 features (CV PR-AUC -0.0007);
the API no longer stores them; promotion gate gains a minimisation clause. Model 0.9.5+xgboost:
test PR-AUC 0.847 (look 3). Subgroup fairness audit with three mitigations."
```

`test_evaluation_look2.json` is the v0.9 model's test result kept before the third look (same
pattern as D3's `test_evaluation_look1.json`). The three D3 figures above were regenerated by the
retrain: they now show the served model; the D3 PDF at tag `v0.3` is unchanged.

## 3. Commit 3 — D5 report and governance documents

```bash
git add \
  reports/src/D5_Optimization_Ethics.md src/boreal_marche/reporting/d5.py \
  src/boreal_marche/reporting/build.py \
  reports/977_D5_Morgunova_OptimizationEthics.md reports/977_D5_Morgunova_OptimizationEthics.pdf \
  reports/figures/d5_latency.png reports/figures/d5_importance.png \
  reports/figures/d5_fairness.png reports/figures/d5_scalability.png \
  GOVERNANCE_LOI25.md MODEL_CARD.md DATA_CARD.md \
  docs/MONITORING.md docs/MLFLOW_PROMOTION_POLICY.md docs/COMMIT_PLAN_D5.md README.md
git diff --cached --stat
git status                                   # expected left over: doc/, .claude/ and the stray file
git commit -m "D5: optimization and ethics report, Law 25 governance, model and data cards" \
  -m "977_D5_Morgunova_OptimizationEthics.pdf (10 pages): latency profile before/after, scalability
breaking points, cost per 1,000 predictions and assistant queries, interpretability assessment,
data-minimisation audit, subgroup fairness audit and mitigations, Law 25 file (CAI guidance
consulted), monitoring policy."
```

**Rebuild only D5.** `python scripts/run.py reports` would also rebuild D1–D4 from the current
evidence (D3 would describe the new model). Use `python -m boreal_marche.reporting.build D5`.

## 4. Push, pull request, merge

```bash
git push -u origin feat/d5
```

Open the PR, wait for the green CI (lint, mypy, 89 tests, Docker build and container smoke test,
gitleaks), merge with **"Create a merge commit"**. Then watch the CD run: the smoke test checks
that `/version` is the merge commit and that the new golden predictions reproduce.

## 5. Evidence follow-up (small second PR, like D4) — then the tag

After CD is green, collect what the first PR could not contain:

```bash
git switch master && git pull --ff-only
git switch -c feat/d5-evidence
aws sso login --profile admin-profile
python scripts/collect_aws_costs.py --profile admin-profile --start 2026-09-01   # one $0.01 API call
python scripts/latency_profile.py --label live_v095 --url https://padp32qm6p.us-east-1.awsapprunner.com -n 40
python -m boreal_marche.reporting.build D5       # "Total cloud spend" is no longer PENDING
git add reports/metrics/aws_costs.json reports/metrics/latency_profile.json \
  reports/977_D5_Morgunova_OptimizationEthics.md reports/977_D5_Morgunova_OptimizationEthics.pdf \
  reports/figures/d5_latency.png reports/figures/d5_scalability.png \
  reports/figures/d5_fairness.png reports/figures/d5_importance.png
git commit -m "D5: total AWS spend from Cost Explorer, live latency of v0.9.5"
git push -u origin feat/d5-evidence            # PR, green CI, merge commit
```

The report table shows the `live_v09` run. The `live_v095` run is stored next to it for D6.

Tag the final merge on `master`:

```bash
git switch master && git pull --ff-only
git log --oneline -6                            # the evidence merge must be on top
git tag -a v0.9.5 -m "D5 — Optimization and Ethics Report (977_D5_Morgunova_OptimizationEthics.pdf)"
git show --stat v0.9.5 | head -15
git push origin v0.9.5
```

If you skip the evidence PR (e.g. SSO is not available before the due date), tag the D5 merge
commit instead: `git tag -a v0.9.5 "$(git log --format=%H -1 --grep='^Merge pull request.*feat/d5')" -m "..."`.
The report then shows the spend as PENDING, which is accurate.

**Windows PowerShell:** same commands, but end continued lines with a backtick instead of `\`, and
write `(git log --format=%H -1 --grep='...')` instead of `"$(git log ...)"`.

## 6. SageMaker (when the quota arrives)

Case 179073448000753 (`ml.m5.large` spot training, `ca-central-1`) was still open on 2026-11-01.
D5 does not depend on it. When it is approved, follow `docs/runbooks/SAGEMAKER_HPO.md`
(`python scripts/run.py hpo --provider aws`). The training entry point already uses the 19 model
features. Then update D3 §11 and mention the run in D6/D7. Do not re-tag `v0.9.5`; the run belongs
to the next tag.
