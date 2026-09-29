# PLAN — Boréal Marché Capstone, D1–D3 execution

Driven by [MASTER_PROMPT_D1_D3.md](MASTER_PROMPT_D1_D3.md). Status legend: `[x]` done · `[~]` done with
a documented limitation · `[ ]` open · **PENDING (Olga)** needs the student.
Executed 2026-09-29. Nothing was committed, pushed or tagged; the git index is untouched.

## Steps

- [x] **0. Environment & hygiene** — `uv` + CPython 3.11.16 `.venv`; CRLF → LF; `.gitattributes`;
  `.gitignore` covers raw/bronze/silver/gold data, MLflow state, challenger artifacts, stray xlsx copies
- [x] **1. Ingestion (bronze)** — `make data`: local copy or UCI download → SHA-256 verified →
  explicit Parquet schema → `data/manifest.json` (no placeholders); DVC initialised, raw file tracked
- [x] **2. Cleaning (silver) + DQ suite** — 11 ordered rules, `reports/cleaning_report.md`;
  `boreal_marche.data.quality` (`make dq`) — caught a real bug (duplicate key), fixed
- [x] **3. Re-skin + EDA** — region/category mapping (`reskin.py`, `DATA_CARD.md`), EDA figures
- [x] **4. Features (gold) + leakage tests** — 37 point-in-time features, 6 quarterly snapshots,
  `reports/feature_spec.md`, `tests/test_features.py`
- [x] **5. Spot-check** — 10 families + dummy + business rule + recency-decile heuristic, identical
  customer-grouped folds
- [x] **6. Optuna tuning** — top-2 (extra trees, XGBoost), 40 TPE trials each, median pruning,
  all trials in MLflow
- [x] **7. Calibration, threshold, artifact, registry, test, error analysis** — sigmoid; profit
  threshold 0.56; `model.joblib` + sidecar; registry aliases; test PR-AUC 0.8504 (2 looks, explained)
- [x] **8. Head B** — sparse CSR (35.5× smaller), Apriori vs FP-Growth timed, sweep, lift trap
  example, `rules.json` with header, indexed engine, hit-rate@5 0.145 vs 0.017 popularity
- [x] **9. API + tests + CI** — `/health /ready /version /predict /predict/batch /recommend`;
  48 tests, 92% coverage; CI: ruff, mypy, pytest + coverage gate, docker build, gitleaks
- [~] **10. SageMaker** — script mode + HPO launcher verified locally; **job not run (PENDING)**
- [x] **11. Docs** — README, ARCHITECTURE, DATA_CARD, MODEL_CARD, GOVERNANCE_LOI25 (design level),
  promotion policy, AWS runbooks, `.env.example`
- [x] **12. Reports** — D1 4 pp., D2 10 pp. (+ `reports/figures/architecture.png`), D3 12 pp.;
  every number rendered from `reports/metrics/*.json` by `make reports`
- [x] **13. Verification** — clean end-to-end re-run (12 min 44 s, exit 0); ruff/mypy clean;
  Docker image builds, runs non-root and reports healthy (2.21 GB — D4 will slim it)

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-09-29 | Reports in English; customer notices EN+FR | User answer; course material is English |
| 2026-09-29 | AWS steps delivered as scripts + runbooks, marked PENDING | Org policy forbids `aws` CLI for the agent; user choice |
| 2026-09-29 | `uv` + CPython 3.11 in `.venv` | `pyproject` requires 3.11; host has 3.14 only |
| 2026-09-29 | Solo charter at 14 h/week | 2-11-5 ponderation = 210 h / 15 weeks |
| 2026-09-29 | Drop the 2009-10 sheet's copy of invoices that also appear in the 2010-11 sheet | Sheets overlap 2010-12-01..09 (22,523 lines) |
| 2026-09-29 | Cancellations separated, never netted | Netting leaks post-cutoff events into pre-cutoff features |
| 2026-09-29 | Observation 180 d, label 90 d, quarterly cut-offs; val 2011-06-01, test 2011-09-01 | All seasons in training; no label-window overlap |
| 2026-09-29 | Cadence on distinct order *days* | Same-day invoices made gaps 0 and `overdue_ratio` explode |
| 2026-09-29 | Added recency-decile heuristic as a third baseline | Shows what ML adds beyond recency |
| 2026-09-29 | Value horizon 365 d (was 90 d) | 90 d made every coupon unprofitable — documented negative result |
| 2026-09-29 | Head B: support 0.005, confidence 0.2, lift ≥ 3, rank by confidence | Sweep; rules/time trade-off; grid extended to 0.003 to check the edge |
| 2026-09-29 | Duplicate key excludes description | DQ suite found 2 duplicates after canonicalisation |
| 2026-09-29 | `sqlalchemy < 2.1` | MLflow 2.22 incompatible with SQLAlchemy 2.1 |
| 2026-09-29 | Unstaged files that `dvc init` added to the git index | Commits are a human decision |
| 2026-09-29 | Pinned cross-platform locks `requirements.txt` / `requirements-dev.txt` (uv `--universal`, tested versions) | Install on another computer, incl. Windows; wheels verified for Windows, macOS arm64, Linux |
| 2026-09-29 | HPO backend selectable: `local` / `aws` / `azure` (`params.yaml → cloud`, `make hpo PROVIDER=`) | User request; SageMaker stays the graded default, Azure needs instructor approval |
| 2026-09-29 | `scripts/run.py` task runner; Makefile delegates to it | Windows has no `make` |
| 2026-09-29 | `evaluate --skip-if-done` in the full pipeline | A fresh clone contains the test guard file; `make all` must not fail there |
| 2026-09-29 | Three commits for the three tags, snapshots verified in scratch dirs | `docs/COMMIT_PLAN_TAGS.md` |

## Hand-off for Olga — do these before committing / submitting

1. **Review the diff** and make sure you can explain every function (course §8.3). Reports are written
   in your voice ("I"); edit anything you would not say at the defence.
2. **Branch, don't commit on `master`:** `git switch -c feat/d1-d3` → three commits (step 9) →
   `git push -u origin feat/d1-d3` → pull request (CI runs lint, types, tests, docker build, secret scan).
3. **Raw data is committed in history** (`online_retail_II.xlsx` at the repo root, commit `a803635`,
   45 MB). Run `git rm --cached online_retail_II.xlsx` in the same PR (the file is now git-ignored).
   It stays in history; before making the repository public consider `git filter-repo` (the licence
   permits redistribution, but the course rule is "never committed").
4. **`doc/`** (course PDF, presentation, local xlsx copy) is untracked; the xlsx is ignored. Decide
   whether the instructor's materials belong in a repository that becomes public.
5. **DVC:** choose a remote (`dvc remote add -d storage s3://…` or a local/Drive remote), `dvc push`,
   then commit `.dvc/`, `.dvcignore`, `data/raw/online_retail_II.xlsx.dvc`.
6. **PENDING — AWS budget alarm ($10)** → `docs/runbooks/AWS_BUDGET_ALARM.md`; insert the screenshot in
   `reports/src/D1_Project_Proposal_Charter.md` §9 and run `make reports`.
7. **PENDING — SageMaker HPO + endpoint + cost** → `docs/runbooks/SAGEMAKER_HPO.md`; fill D3 §11.
   Azure ML is prepared as an alternative (`docs/runbooks/AZURE_ML_SWEEP.md`) — only with the
   instructor's approval, since the specification names SageMaker.
8. **PENDING — Law 25 primary source:** verify the article summaries in `GOVERNANCE_LOI25.md` against
   LégisQuébec / CAI guidance and record URLs + dates (the assignment requires reading the source).
9. **Commits and tags** `v0.1`, `v0.2`, `v0.3`: follow [docs/COMMIT_PLAN_TAGS.md](docs/COMMIT_PLAN_TAGS.md)
   (exact `git add` lists; each snapshot was verified to pass its own CI; merge with a merge commit).
10. **Sidecar commit hash:** `model_meta.json` records `a803635…-dirty` (trained on the uncommitted
    tree). Either keep it and say so, or re-run `make train` and `evaluate --force` after committing —
    that would be a third test look, which the report must then state.
11. Add the instructor as a collaborator; enable branch protection requiring green CI.
12. PDFs are named `977_D<n>_Morgunova_<Deliverable>.pdf` (§7.1); submit them on the course platform.

## Running on another computer

Windows from zero: [docs/RUN_FROM_ZERO_WINDOWS.md](docs/RUN_FROM_ZERO_WINDOWS.md). A fresh-clone
reproduction (only the files that will be committed, new venv from `requirements-dev.txt`, workbook
downloaded from UCI) was run end to end on Linux on 2026-09-29.

## Environment notes

`uv` was installed to `~/.local/bin/uv` and CPython 3.11 to `~/.local/share/uv/python/`; the project
environment is `.venv/`. Regenerate everything with `make setup && make all && make test && make reports`.
