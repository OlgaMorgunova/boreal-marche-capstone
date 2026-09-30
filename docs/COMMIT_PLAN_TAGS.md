# Commit and tag plan — `v0.1` (D1), `v0.2` (D2), `v0.3` (D3)

The course requires an annotated tag per deliverable (§7.1: v0.1 D1 · v0.2 D2 · v0.3 D3). All three
were implemented in one working session, so the working tree is split into **three sequential
commits**, each a self-consistent snapshot of what existed at that deliverable.

**Verified before writing this file:** each snapshot was rebuilt in a scratch directory and checked
with the CI that is active *at that commit* — v0.1: `ruff check .` + 12 tests pass (3 real-data tests
skip, as in CI); v0.2: 30 tests; v0.3: the new CI (ruff, format, mypy strict, 57 tests, coverage 91%
≥ 70%). The v0.1/v0.2 snapshots were tested with mlflow/optuna/matplotlib/mlxtend *blocked*, because
the old workflow installs only `.[dev]`.

## Rules the split follows

- `src/boreal_marche/api.py`, `.github/workflows/ci.yml`, `README.md` stay **unstaged until commit 3**:
  the new API needs Head A/B code, and the new CI builds the Docker image, which needs the artifacts.
  Commits 1–2 therefore keep the original versions of these three files (they still pass).
- Everything a report cites is committed with that report (its metrics JSON and figures).
- Not committed at all: the raw workbook (DVC), `data/bronze|silver|gold`, `mlflow.db`, `mlruns/`,
  challenger artifacts, `reports/*.html` — all git-ignored. **`doc/`** (the instructor's PDF,
  presentation, `db_path.txt`) is left out of this plan on purpose: decide whether course material
  belongs in a repository that becomes public (add it in any commit if yes).

## 0. Branch (org rule: never commit on a protected branch, never track it)

```bash
cd boreal-marche-capstone
git status                                   # nothing staged yet
git switch -c feat/d1-d3                     # created from local HEAD: no upstream is set
git branch -vv                               # must NOT show [origin/master] for feat/d1-d3
```

## 1. Commit 1 → tag `v0.1` — D1 Project Proposal and Charter

Content: repository hygiene, pinned requirements, `params.yaml`, reproducible ingestion
(SHA-256, explicit schema, manifest), DVC, cleaning + report, data-quality suite, re-skin, the
feature module used by the DQ suite, report tooling, the D1 report. Works at this tag:
`make data etl features dq` / `python scripts/run.py data etl features dq`.

```bash
git rm --cached online_retail_II.xlsx        # raw data out of Git (file stays on disk, now ignored)
git add \
  .gitattributes \
  .gitignore \
  .env.example \
  pyproject.toml \
  requirements.txt \
  requirements-dev.txt \
  params.yaml \
  Makefile \
  scripts/run.py \
  .dvc/config \
  .dvc/.gitignore \
  .dvcignore \
  data/raw/online_retail_II.xlsx.dvc \
  data/manifest.json \
  src/boreal_marche/__init__.py \
  src/boreal_marche/config.py \
  src/boreal_marche/utils.py \
  src/boreal_marche/reskin.py \
  src/boreal_marche/features.py \
  src/boreal_marche/data/__init__.py \
  src/boreal_marche/data/ingest.py \
  src/boreal_marche/data/cleaning.py \
  src/boreal_marche/data/quality.py \
  src/boreal_marche/reporting/__init__.py \
  src/boreal_marche/reporting/style.py \
  src/boreal_marche/reporting/figures.py \
  src/boreal_marche/reporting/build.py \
  scripts/checksum.py \
  scripts/download_data.py \
  scripts/etl.py \
  scripts/data_quality.py \
  tests/__init__.py \
  tests/conftest.py \
  tests/test_health.py \
  tests/test_ingest.py \
  tests/test_cleaning.py \
  tests/test_data_quality.py \
  MASTER_PROMPT_D1_D3.md \
  PLAN.md \
  DATA_CARD.md \
  docs/runbooks/AWS_BUDGET_ALARM.md \
  reports/src/D1_Project_Proposal_Charter.md \
  reports/977_D1_Morgunova_ProjectProposal.md \
  reports/977_D1_Morgunova_ProjectProposal.pdf \
  reports/cleaning_report.md \
  reports/metrics/cleaning_report.json \
  reports/figures/architecture.png \
  reports/figures/eda_overview.png \
  reports/figures/cleaning_rules.png
git diff --cached --stat                     # review: api.py, ci.yml, README.md must NOT be listed
git commit -m "D1: project proposal and charter, reproducible ingestion and cleaning" \
  -m "UCI Online Retail II ingestion with SHA-256 verification, explicit Parquet schema and manifest;
DVC-tracked raw data; 11 documented cleaning rules with report; data-quality suite; re-skin mapping;
pinned cross-platform requirements; D1 report (977_D1_Morgunova_ProjectProposal.pdf)."
```

## 2. Commit 2 → tag `v0.2` — D2 Model Design and Architecture

Content: point-in-time feature tests, shared preprocessing and model zoo, customer-grouped CV,
spot-check, Optuna search code, profit and artifact classes, MLflow helpers, architecture and
governance design, the D2 report. Works at this tag: + `spotcheck`, `tune`.

```bash
git add \
  src/boreal_marche/tracking.py \
  src/boreal_marche/retention/__init__.py \
  src/boreal_marche/retention/models.py \
  src/boreal_marche/retention/cv.py \
  src/boreal_marche/retention/spotcheck.py \
  src/boreal_marche/retention/tune.py \
  src/boreal_marche/retention/artifact.py \
  src/boreal_marche/retention/profit.py \
  tests/test_features.py \
  tests/test_retention.py \
  ARCHITECTURE.md \
  GOVERNANCE_LOI25.md \
  reports/src/D2_Design_Architecture.md \
  reports/977_D2_Morgunova_DesignArchitecture.md \
  reports/977_D2_Morgunova_DesignArchitecture.pdf \
  reports/feature_spec.md \
  reports/metrics/gold_summary.json \
  reports/metrics/spotcheck.json \
  reports/figures/windows.png \
  reports/figures/spotcheck_boxplot.png
git diff --cached --stat
git commit -m "D2: model design and architecture, spot-check of 10 model families" \
  -m "Feature specification (37 point-in-time features) with leakage tests; identical-fold spot-check
against three baselines; Optuna search spaces; profit model; architecture diagram and Law 25
design; D2 report (977_D2_Morgunova_DesignArchitecture.pdf)."
```

## 3. Commit 3 → tag `v0.3` — D3 Model Implementation and Validation

Content: training/calibration/threshold, one-time test evaluation, leakage study, Head B mining and
serving engine, typed API, MLflow registry resolution, SageMaker/Azure/local HPO backends, artifacts,
Docker, the full CI, cards, runbooks, the D3 report. Works at this tag: everything
(`make all`, `make test`, `make reports`, `make up`).

```bash
git add \
  src/boreal_marche/api.py \
  src/boreal_marche/schemas.py \
  src/boreal_marche/registry.py \
  src/boreal_marche/retention/train.py \
  src/boreal_marche/retention/evaluate.py \
  src/boreal_marche/retention/leakage.py \
  src/boreal_marche/retention/sm_train.py \
  src/boreal_marche/recommend/__init__.py \
  src/boreal_marche/recommend/baskets.py \
  src/boreal_marche/recommend/mine.py \
  src/boreal_marche/recommend/engine.py \
  src/boreal_marche/recommend/evaluate.py \
  src/boreal_marche/cloud/__init__.py \
  src/boreal_marche/cloud/__main__.py \
  src/boreal_marche/cloud/common.py \
  src/boreal_marche/cloud/aws.py \
  src/boreal_marche/cloud/azure.py \
  tests/test_engine.py \
  tests/test_api.py \
  tests/test_pipeline_smoke.py \
  tests/test_cloud.py \
  tests/golden/golden_predictions.json \
  artifacts/model.joblib \
  artifacts/model_meta.json \
  artifacts/rules.json \
  docker/Dockerfile \
  docker/compose.yml \
  .dockerignore \
  .github/workflows/ci.yml \
  README.md \
  MODEL_CARD.md \
  docs/MLFLOW_PROMOTION_POLICY.md \
  docs/CLOUD_OPTIONS.md \
  docs/RUN_FROM_ZERO_WINDOWS.md \
  docs/COMMIT_PLAN_TAGS.md \
  docs/runbooks/SAGEMAKER_HPO.md \
  docs/runbooks/AZURE_ML_SWEEP.md \
  reports/src/D3_Implementation_Validation.md \
  reports/977_D3_Morgunova_ImplementationValidation.md \
  reports/977_D3_Morgunova_ImplementationValidation.pdf \
  reports/metrics/tuning.json \
  reports/metrics/train_final.json \
  reports/metrics/test_evaluation.json \
  reports/metrics/test_evaluation_look1.json \
  reports/metrics/leakage.json \
  reports/metrics/rules_sweep.json \
  reports/metrics/rules_mining.json \
  reports/metrics/rules_evaluation.json \
  reports/figures/optuna_history_extra_trees.png \
  reports/figures/optuna_history_xgboost.png \
  reports/figures/optuna_importance_extra_trees.png \
  reports/figures/optuna_importance_xgboost.png \
  reports/figures/reliability_validation.png \
  reports/figures/reliability_test.png \
  reports/figures/profit_curve_validation.png \
  reports/figures/error_segments.png \
  reports/figures/rules_sweep.png
git diff --cached --stat
git commit -m "D3: model implementation and validation, rule engine, typed API, cloud HPO backends" \
  -m "Tuned + calibrated XGBoost (test PR-AUC 0.850), profit threshold, leakage investigation,
FP-Growth rules with indexed serving (hit-rate@5 0.145 vs 0.017), FastAPI endpoints with contract
and golden tests, MLflow registry aliases, SageMaker/Azure ML/local HPO launchers, Docker, CI;
D3 report (977_D3_Morgunova_ImplementationValidation.pdf)."
git status                                   # expected: clean, except doc/ (see above)
```

## 4. Push, pull request, merge

```bash
git push -u origin feat/d1-d3                # first push: explicit destination (org rule)
git rev-parse --abbrev-ref feat/d1-d3@{upstream}   # must print origin/feat/d1-d3
```

Open the PR on GitHub, wait for the green CI, then merge with **"Create a merge commit"** — not
*Squash* (would collapse the three commits into one) and not *Rebase* (would change their SHAs and
orphan the tags).

## 5. Tags (annotated), after the merge

```bash
git switch master && git pull --ff-only
git log --oneline -5                         # find the three commits
git tag -a v0.1 "$(git log --format=%H -1 --grep='^D1:')" -m "D1 — Project Proposal and Charter (977_D1_Morgunova_ProjectProposal.pdf)"
git tag -a v0.2 "$(git log --format=%H -1 --grep='^D2:')" -m "D2 — Model Design and Architecture (977_D2_Morgunova_DesignArchitecture.pdf)"
git tag -a v0.3 "$(git log --format=%H -1 --grep='^D3:')" -m "D3 — Model Implementation and Validation (977_D3_Morgunova_ImplementationValidation.pdf)"
git show --stat v0.1 | head -20              # check each tag points where expected
git push origin v0.1 v0.2 v0.3
```

Windows PowerShell: identical commands, except replace `"$(git log ...)"` with
`(git log --format=%H -1 --grep='^D1:')` and put each `git add` list on one line or end the lines
with a backtick instead of `\`.

## 6. After tagging

- `artifacts/model_meta.json` records the commit it was trained on as `a803635…-dirty` (the tree
  before these commits). Keep it and mention it, or retrain after the merge (`make train` then
  `make evaluate ARGS=--force`); the latter adds a **third test look**, which D3 §8 must then state.
- Tick the tags in `README.md` → *Deliverables* and in `PLAN.md` → *Hand-off*.
- Later deliverables continue the same pattern: `v0.9` (D4), `v1.0` (D7).
