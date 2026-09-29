# MASTER PROMPT — Boréal Marché Capstone, Deliverables D1–D3

> **Purpose of this file.** This is the corrected, fully specified version of the original request
> (preserved verbatim in Appendix Z). It is written to be executed by an AI coding agent
> (Claude Code or equivalent) step by step, and to be re-run or audited by a human reviewer.
> Execution status is tracked in [PLAN.md](PLAN.md).

---

## 0. Role and operating mode

You are a **senior ML engineer and technical writer** acting as a pair-programmer for the student
**Olga Morgunova** (individual/solo project, course 420-977-VA, Vanier College, LEA.EU).
You produce code, artifacts and reports **on her behalf**, in her repository, to the standard of the
course rubric's *Excellent (90–100%)* band.

Operating rules (non-negotiable):

1. **Honesty over polish.** Every number in a report must be produced by code in this repository and
   be reproducible with a documented command. Never invent a metric, screenshot, cost, or result.
   If something could not be run (e.g., AWS), say so and mark it `PENDING — requires Olga`.
   A documented negative result is worth marks (Honesty bonus, §6.4 of the assignment).
2. **No commits, no pushes, no tags.** Leave all changes in the working tree. The human reviews and
   commits. Do not rewrite git history and do not modify the git index.
3. **No AWS CLI / Terraform execution** (organisation policy). Cloud steps are delivered as scripts +
   runbooks that Olga executes manually.
4. **Business logic lives in `src/`**, not in notebooks or ad-hoc scripts. Scripts and Make targets
   are thin wrappers.
5. **Configuration is declared, not buried**: observation/label windows, cut-offs, cleaning
   parameters, cost assumptions and seeds live in `params.yaml`.
6. **Raw data is never committed to Git.** It is fetched/verified by one command and tracked by DVC.
7. **Reports are written in English** (decision 2026-09-29); customer-facing notices are EN + FR.
8. Ask the user only for decisions that genuinely belong to them; otherwise choose the conventional
   default, record it in §9 (Decision log) and proceed.

---

## 1. Context

### 1.1 Inputs available in the repository

| Input | Path | Notes |
|---|---|---|
| Assignment specification (31 pp.) | `doc/Capstone_Project1.pdf` | Source of truth for requirements and rubrics |
| Kick-off presentation | `doc/Capstone_Project1_Presentation.html` | Framing: "mining is expensive, lookup is cheap", profit threshold, golden set |
| Dataset (local copy) | `doc/online_retail_II.xlsx` (also a committed copy at repo root) | UCI Online Retail II, 2 sheets (2009-2010, 2010-2011) |
| Dataset canonical URL | `doc/db_path.txt` → `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip` | DOI 10.24432/C5CG6D, licence CC BY 4.0 |
| Existing skeleton | `src/`, `scripts/`, `tests/`, `docker/`, `.github/workflows/ci.yml`, `*.md` cards | Needs correction (see §1.3) |

### 1.2 Client scenario (from the assignment)

Boréal Marché — mid-size Québec online grocery/specialty retailer (~5,000 customers, ~4,000 active SKUs,
~25,000 orders/year; QC, ON, Maritimes). Three problems run on intuition:

| Head | Technique | Business question |
|---|---|---|
| A — Prédiction | Supervised binary classification | Will this customer place another order in the next 90 days? |
| B — Recommandation | Association rules (Apriori / FP-Growth) | Given this basket, what should we suggest next? |
| C — Assistant | RAG + tool calling, 3 providers | Answer customers in FR/EN, grounded in policy and in our own models |

### 1.3 Known defects in the existing skeleton (must be fixed)

| # | Defect | Fix |
|---|---|---|
| 1 | `scripts/data_quality.py` uses `InvoiceNo`, `UnitPrice`; the real columns are `Invoice`, `Price`, `Customer ID` | Rewrite as a pandera-free assertion suite in `src/` + pytest |
| 2 | `data/manifest.json` has `"sha256": "CALCULATED_VALUE"` | Compute and record real SHA-256, row count, retrieval date, licence |
| 3 | `download_data.py` saves a zip but `etl.py` expects an xlsx; no checksum verification; no explicit schema | Single `make data` command: acquire → verify SHA-256 → Parquet with explicit schema → manifest |
| 4 | Scripts have CRLF line endings (only change in the working tree) | Normalise to LF; add `.gitattributes` |
| 5 | Raw xlsx committed to Git at repo root (45 MB) | `.gitignore` it; DVC-track `data/raw`; hand-off note for Olga to `git rm --cached` |
| 6 | Makefile targets point to non-existent modules (`boreal_marche.etl`, `.rules`, `.train`…) | Implement the modules |
| 7 | `config.py` only holds `label_window_days` | Typed config loaded from `params.yaml` |
| 8 | CI: no type check, no coverage gate, no data-quality suite, no secret scan, no docker build | Extend `ci.yml` (full CD is D4) |
| 9 | Dockerfile copies tests into image, single stage, unpinned base, `pip install -e` | Record as D4 work; minimal fix only if needed for tests |
| 10 | Python pinned to 3.11 but host has only 3.14 | Use `uv` to provision CPython 3.11 in `.venv` |

### 1.4 Environment facts

- Host: Linux, 16 cores, ~30 GB RAM, no GPU needed. Python via `uv` → `.venv` (CPython 3.11).
- Tools available: Docker, headless Chrome (HTML→PDF), LibreOffice, `pdfinfo`.
- MLflow runs locally with a SQLite backend (`sqlite:///mlflow.db`, artifacts in `mlruns/`).

---

## 2. Objective and scope

Deliver **D1, D2 and D3** of the assignment, each as (a) the supporting code/artifacts in the repo and
(b) a written report in Markdown **and** PDF under `reports/`, named per §7.1 of the assignment:

| Deliverable | Report file | Length target | Repo tag (Olga creates) | KLO |
|---|---|---|---|---|
| D1 Project Proposal and Charter | `reports/977_D1_Morgunova_ProjectProposal.pdf` | 4–6 pp. | `v0.1` | 1 |
| D2 Model Design and Architecture | `reports/977_D2_Morgunova_DesignArchitecture.pdf` + `reports/figures/architecture.png` | 10–14 pp. | `v0.2` | 2 |
| D3 Model Implementation and Validation | `reports/977_D3_Morgunova_ImplementationValidation.pdf` | 12–18 pp. | `v0.3` | 3 |

**In scope:** data pipeline (bronze/silver/gold), Head A end-to-end (features → spot-check → Optuna →
calibration → profit threshold → test → error analysis → artifact + sidecar → MLflow registry),
Head B end-to-end (sparse baskets → Apriori vs FP-Growth → sweep → lift filter → `rules.json` →
indexed serving engine → hit-rate@5/coverage vs popularity), SageMaker-compatible training script and
HPO launcher (not executed), typed API endpoints for `/health`, `/ready`, `/version`, `/predict`,
`/predict/batch`, `/recommend`, tests, CI improvements, updated cards and architecture docs.

**Out of scope for this run (later deliverables):** Head C implementation (corpus, index, providers,
golden set — Week 9–10), CD/OIDC, cloud deployment, Streamlit UI, monitoring dashboard, fairness
audit & final Law 25 file (D5), executive summary/presentation (D6). D2 still contains the *design*
of all of these.

---

## 3. Global requirements traceability (assignment §3 → where satisfied)

| Req. | Requirement (abridged) | Where |
|---|---|---|
| 3.1 | Real data with multi-item transactions, timestamps, customer id | D1 §Data, `DATA_CARD.md` |
| 3.1 | One command: acquire → SHA-256 verify → Parquet explicit schema → manifest | `make data` → `boreal_marche.data.ingest` |
| 3.1 | DVC-tracked raw data | `dvc add data/raw/online_retail_II.xlsx` |
| 3.1 | Honest provenance + documented re-skin mapping | `DATA_CARD.md`, `boreal_marche.data.reskin` |
| 3.2 | Windows declared in config | `params.yaml: windows` |
| 3.2 | Cleaning with before/after report, rows affected + rationale per rule | `reports/cleaning_report.{md,json}` |
| 3.2 | ≥25 documented features, explicit missingness | `boreal_marche.features`, `reports/feature_spec.md` |
| 3.2 | ≥8 model families, identical folds & preprocessing, 2 baselines, PR-AUC/ROC-AUC/Brier/recall@top-decile/fit time | `boreal_marche.retention.spotcheck` |
| 3.2 | Optuna (TPE + pruning) on top-2, all trials in MLflow | `boreal_marche.retention.tune` |
| 3.2 | Calibration before/after + profit-maximising threshold with stated costs | `boreal_marche.retention.calibrate` |
| 3.2 | Error analysis by segment | `boreal_marche.retention.evaluate` |
| 3.3 | Sparse basket matrix + dense vs sparse memory | `boreal_marche.recommend.baskets` |
| 3.3 | Apriori (timed) and FP-Growth (production), support/confidence sweep | `boreal_marche.recommend.mine` |
| 3.3 | Lift filter + real high-confidence/low-lift example | D3 §Head B |
| 3.3 | `rules.json` with header (date, params, data hash, rule count) | `artifacts/rules.json` |
| 3.3 | Indexed serving engine, dedupe by consequent (max lift), exclude basket items, return firing rule, empty/no-match strategies | `boreal_marche.recommend.engine` |
| 3.3 | Hit-rate@5 & coverage on temporal holdout vs popularity | `boreal_marche.recommend.evaluate` |
| 3.5 | Script-mode training (`SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`), HPO launcher, cost comparison | `boreal_marche.retention.sm_train`, `boreal_marche.cloud` (local/aws/azure), runbooks |
| 3.5 | Single joblib artifact (preprocessing + estimator + calibrator) + metadata sidecar | `artifacts/model.joblib`, `artifacts/model_meta.json` |
| 3.5 | MLflow tracking + registry + written promotion policy, alias resolution with local fallback | `docs/MLFLOW_PROMOTION_POLICY.md`, `boreal_marche.registry` |

---

## 4. Technical design decisions (defaults to implement)

### 4.1 Data layers

- **Bronze** `data/bronze/transactions.parquet` — both sheets concatenated verbatim, explicit schema
  (`invoice: string, stock_code: string, description: string, quantity: int32, invoice_date: timestamp,
  price: float64, customer_id: Int64 (nullable), country: string, source_sheet: string`), plus
  `row_id`. Only type coercion, no row removal.
- **Silver** `data/silver/transactions_clean.parquet` — cleaning rules applied in a fixed order; each
  rule logs rows before/after and rationale. Two views: `lines_head_a` (requires customer id) and
  `lines_head_b` (guest invoices allowed).
- **Gold** `data/gold/retention_{split}.parquet` — one row per (customer, cutoff) snapshot, point-in-time.

### 4.2 Cleaning rules (order matters; each reports rows affected)

1. Drop exact duplicate lines.
2. Separate cancellations (`invoice` starts with `C`) → kept in a `cancellations` table *used only as a
   point-in-time feature source*, never to retroactively delete earlier invoices (that would leak future
   information into past snapshots).
3. Drop non-positive quantity (non-cancellation adjustments).
4. Drop non-positive price (free samples, adjustments `A` invoices, damages).
5. Drop service / non-product SKUs (`POST`, `DOT`, `M`, `D`, `C2`, `BANK CHARGES`, `AMAZONFEE`, `S`,
   `B`, `CRUK`, `PADS`, `ADJUST*`, `TEST*`, gift vouchers…) — pattern list in `params.yaml`.
6. Canonicalise descriptions: strip/upper-case/collapse whitespace; one canonical description per
   `stock_code` = most frequent variant.
7. Outlier treatment: flag lines with quantity or line revenue above the 99.9th percentile; cap for
   monetary features (winsorise) rather than drop, and drop the extreme bulk lines known to be
   immediately cancelled (e.g., the 80,995 / 74,215 unit errors) only via rule 2/3 logic.
8. Null customer id: removed **for Head A only** (cannot attribute to a customer), retained for Head B
   baskets. Row count reported for both.

### 4.3 Windows (Head A)

- Label window **90 days**: `y = 1` if the customer has ≥1 valid invoice in `[cutoff, cutoff + 90d)`.
- Observation window **180 days**: customer is *eligible* at a cutoff if they have ≥1 valid invoice in
  `[cutoff − 180d, cutoff)`; windowed features use this range; lifetime features (tenure, lifetime
  orders) use all history `< cutoff` and are flagged as left-censored at 2009-12-01.
- Cut-offs (quarterly, no label-window overlap across splits):
  - **train**: 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01
  - **validation** (calibration + threshold): 2011-06-01
  - **test** (touched once): 2011-09-01 (label window ends 2011-11-30 < data end 2011-12-09)
- Spot-check CV: 5-fold `StratifiedGroupKFold` on the stacked train snapshots, grouped by customer id,
  fixed seed — identical folds for every model.

### 4.4 Feature table (≥25 features, typed, missingness explicit)

Families: recency (days since last order, days since first order), frequency (orders 30/90/180 d,
lifetime), monetary (revenue 30/90/180 d, mean/median/max order value), cadence (mean/std inter-order
gap, gap trend, days since last minus mean gap), basket shape (mean lines per order, mean units per
line, distinct SKUs, SKU repeat rate), category mix (share of top keyword categories, category
entropy), behavioural (cancellation count & rate, weekend share, evening share, month-of-first-order
seasonality flag, trend of spend first vs second half of window), geography (re-skinned region).
Undefined values (e.g., inter-order gap for one-order customers) are `NaN` + explicit indicator, never
silently `0`. Spec written to `reports/feature_spec.md` (name, family, dtype, definition, rationale,
missingness policy).

### 4.5 Modelling protocol (Head A)

- Shared preprocessing `ColumnTransformer`: numeric → `SimpleImputer(median, add_indicator=True)` →
  `StandardScaler`; categorical → `OneHotEncoder(handle_unknown="ignore")`.
- Baselines: `DummyClassifier(strategy="prior")` and one-rule business heuristic
  (rank by recency; binary rule "ordered in last 60 days").
- ≥8 families: Logistic Regression, Gaussian NB, k-NN, Decision Tree, Random Forest, Extra Trees,
  HistGradientBoosting, LightGBM, XGBoost, MLP.
- Metrics: **PR-AUC (primary)**, ROC-AUC, Brier, recall@top-decile, precision@top-decile, fit time;
  mean ± std across folds; box-plot of fold PR-AUC.
- Optuna: `TPESampler(seed)` + `MedianPruner`, fold-level `trial.report` for pruning, ≥40 trials per
  model, all trials logged to MLflow (nested runs); report search space, history, importances,
  improvement over defaults.
- Calibration: compare none / sigmoid / isotonic via out-of-fold Brier on the validation snapshot;
  reliability curves before/after; final calibrator fit on full validation snapshot.
- Threshold: action = send retention coupon to customers at risk (`risk = 1 − p_return`).
  Profit per contacted customer (stated assumptions in `params.yaml`):
  `lapser saved: save_rate × customer_value − coupon_cost`;
  `would-return anyway: − (coupon_cost + discount_rate × median_order_value × gross_margin_leak)`.
  Choose the threshold maximising realised profit on validation; compare against *coupon to everyone*
  (status quo) and *coupon to nobody*; sensitivity to `save_rate`.
- Test set: evaluated **once** by `make evaluate`; a guard file records the evaluation; bootstrap 95% CI.
- Leakage investigation: (i) unit test asserting every feature source row has `invoice_date < cutoff`;
  (ii) deliberately leaky variant (features computed through the label window) to show the metric the
  tests would catch; (iii) cancellation-netting leakage explained; (iv) snapshot overlap check.

### 4.6 Head B protocol

- Baskets: silver Head-B lines grouped by invoice → `scipy.sparse.csr_matrix` (invoice × stock_code);
  report dense vs sparse bytes.
- Temporal holdout: mine on invoices `< 2011-09-01`; evaluate on invoices `≥ 2011-09-01` with ≥2 items.
- Apriori (timed) vs FP-Growth (timed, production) on identical parameters (mlxtend).
- Sweep `min_support × min_confidence`; record rule count, runtime, hit-rate@5 on a *validation slice*
  of the mining period (last 2 months before holdout) — the holdout is used once for the final number.
- Lift filter `lift > 1` (+ minimum lift parameter); report a real high-confidence / low-lift rule.
- `artifacts/rules.json` header: `mined_at`, `params`, `source_sha256`, `n_baskets`, `n_rules`, code version.
- Engine: antecedent-indexed dict (frozenset → rules), subset enumeration up to `max_len`, dedupe by
  consequent keeping max lift, exclude basket items, returns `strategy ∈ {rules, popularity_fallback,
  empty_basket}` and the firing rule for explanation.
- Evaluation: leave-one-out hit-rate@5 and coverage (share of baskets served by rules; catalogue
  coverage) vs global-popularity baseline.

### 4.7 MLOps

- MLflow: experiment per head; every run logs params, metrics, data hash, git SHA, plots.
- Registry: model `boreal-retention`, aliases `challenger` → `champion` per written promotion policy.
  Serving resolves `models:/boreal-retention@champion`; falls back to `artifacts/model.joblib`.
- Artifact: one joblib containing preprocessing + estimator + calibrator (`RetentionModel` class in
  `src/`); sidecar `model_meta.json` (version, trained_at, n_rows, feature order, library versions,
  git commit, MLflow run id, threshold, test PR-AUC).
- SageMaker: `boreal_marche.retention.sm_train` uses the script-mode contract and runs locally;
  `python -m boreal_marche.cloud --provider local|aws|azure` builds a Bayesian HPO job (SageMaker
  managed spot in `ca-central-1`, Azure ML low-priority in `canadacentral`, or local Optuna) with
  `max_run`/`max_wait` caps. Not executed here — runbook `docs/runbooks/SAGEMAKER_HPO.md`.
- Budget: `docs/runbooks/AWS_BUDGET_ALARM.md` ($10 alarm, console steps, evidence to capture).

---

## 5. Step-by-step execution plan

Each step has **inputs → actions → outputs → acceptance criteria**. Mark progress in PLAN.md.

### Step 0 — Environment and repo hygiene
- Provision Python 3.11 with `uv`, create `.venv`, install deps from `pyproject.toml`.
- Normalise CRLF → LF, add `.gitattributes`, extend `.gitignore` (`/online_retail_II.xlsx`,
  `doc/*.xlsx`, `data/bronze|silver|gold`, `mlflow.db`, `mlruns/`, large artifacts).
- **Accept:** `make test` and `make lint` run.

### Step 1 — Reproducible ingestion (bronze)
- `make data`: locate source (download zip from UCI or local `doc/` copy) → verify SHA-256 → read both
  sheets with explicit dtypes → bronze Parquet → `data/manifest.json` (source, DOI, URL, licence,
  retrieval date, sha256 of xlsx, sheet row counts, total rows, schema).
- `dvc init` + `dvc add data/raw/online_retail_II.xlsx` (local cache only; remote is Olga's choice).
- **Accept:** second run is idempotent; tampered file fails checksum; manifest has no placeholders.

### Step 2 — Cleaning (silver) + data-quality suite
- Implement rules §4.2; write `reports/cleaning_report.{json,md}`; implement DQ assertions in
  `boreal_marche.data.quality` + `tests/test_data_quality.py` (skipped in CI if data absent, synthetic
  fixture tests always run).
- **Accept:** every rule lists rows affected + rationale; totals reconcile.

### Step 3 — Re-skin + EDA figures
- Country → region mapping (documented), keyword category map, currency label; EDA figures.
- **Accept:** `DATA_CARD.md` states plainly that data are public UCI data re-skinned for pedagogy.

### Step 4 — Features (gold) + leakage tests
- **Accept:** ≥25 features; `tests/test_features.py` proves point-in-time correctness on synthetic data.

### Step 5 — Spot-check
- **Accept:** table of ≥10 models + 2 baselines, fold box-plot, 3-sentence verdict naming top-2.

### Step 6 — Optuna tuning (top-2) with MLflow
### Step 7 — Calibration, threshold, final artifact, registry, one-time test evaluation, error analysis
### Step 8 — Head B mining, engine, evaluation, `rules.json`
### Step 9 — API endpoints + tests + CI
### Step 10 — SageMaker script-mode + HPO launcher + runbooks (not executed)
### Step 11 — Documentation: README, ARCHITECTURE, DATA_CARD, MODEL_CARD, GOVERNANCE_LOI25 (design level), promotion policy
### Step 12 — Reports D1, D2, D3 (Markdown → HTML → PDF via headless Chrome), page counts verified
### Step 13 — Verification pass: re-run pipeline from clean, cross-check every number in reports
against `reports/*.json`, run tests/lint/mypy, write hand-off checklist for Olga.

---

## 6. Report content requirements (rubric-driven)

### D1 (100 pts): Problem framing 20 · Data justification 20 · Architecture sketch 15 · Success metrics 15 · Risk register 15 · Plan & charter 15
Must include: business problem in client language with three distinct sub-problems each tied to a
measurable cost/opportunity; dataset vs three-part criterion + licence + size + known quality issues +
provenance; offline/online architecture sketch with heads & artifacts; technical and business metrics
with targets and a named baseline each (Appendix B targets); explicit out-of-scope; ≥6 risks incl. data
quality, cloud cost, scope creep (likelihood/impact/mitigation/owner); solo charter with 14 h/week
budget, review process, cadence; 15-week plan mapped to §4 milestones; cost plan (ceiling $25, target
<$12) and $10 budget alarm (**evidence PENDING — Olga**); AI-tool use statement.

### D2 (100 pts): Architecture 20 · Data pipeline 20 · Features 15 · Algorithm rationale 20 · Evaluation design 15 · Ethics-by-design 10
Must include: full diagram (separate PNG) + walkthrough; bronze/silver/gold; cleaning table with rows
and rationale; observation/label window diagram; complete feature spec; rationale for all three heads
against named alternatives and when the choice would change (A: families; B: rules vs CF; C: RAG vs
fine-tuning); evaluation design (primary metric, baselines, test protection); ethics-by-design
(automated decision points, Law 25 s.12.1 obligations, design response incl. human approval step);
technology decision table; spot-check table + box-plot + three-sentence verdict.

### D3 (100 pts): Leakage 20 · Spot-check 15 · Tuning 15 · Calibration & threshold 15 · Head B 15 · Cloud & MLOps 10 · Artifact discipline 10
Must include: cleaning report; leakage investigation with before/after metric; tuning (space, trials,
history, importances, improvement); calibration curves + method justification; profit curve + costs +
threshold + business consequence; final test results once with CI; error analysis; Head B sweep,
chosen params, rule count, lift-trap example, hit-rate@5 + coverage vs popularity; SageMaker section
(evidence PENDING, comparison table template with local numbers filled, crossover analysis); MLflow
evidence (run count, comparison, promotion policy); artifact + sidecar + joblib vs pickle risks and
why preprocessing lives inside the artifact; limitations; AI-tool attribution.

Every report: title page (student, course, instructor, date, version tag), captions on every figure,
figures referenced in text, baselines next to every metric, a *Limitations* section, *Sources and
AI-tool use* section, and a reproducibility box with the exact commands.

---

## 7. Quality gates (Definition of Done)

- [ ] `make data etl features spotcheck tune train evaluate rules` runs end-to-end from a clean clone + source file.
- [ ] `pytest` green, coverage ≥ 70% on `src/` (excluding SageMaker launcher), `ruff check` clean, `mypy src` clean or justified ignores.
- [ ] Every number in D1–D3 traceable to a JSON/CSV file under `reports/` or `artifacts/`.
- [ ] PDFs exist, page counts within ranges, figures captioned.
- [ ] No secrets, no raw data in the working tree diff; `.gitignore` covers generated data.
- [ ] PENDING items listed in PLAN.md hand-off section.

---

## 8. Questions to the user (asked and answered 2026-09-29)

| Question | Answer |
|---|---|
| Report language | English (customer notices EN+FR) |
| AWS (budget alarm, SageMaker HPO) | Prepare scripts + checklist; Olga executes manually; reports mark as PENDING |
| Python runtime | `uv` + CPython 3.11 in `.venv` |
| Weekly time budget (solo) | 14 h/week (2-11-5 ponderation: 210 h / 15 weeks) |

Still required from Olga (cannot be decided by the agent): AWS account evidence (budget alarm
screenshot, SageMaker job ARN, cost), instructor added as collaborator, choice of DVC remote, final
review of the Law 25 primary-source reading (Commission d'accès à l'information).

---

## 9. Decision log

Recorded in [PLAN.md](PLAN.md#decision-log) as decisions are made during execution.

---

## Appendix Z — Original request (verbatim)

```text
Откорректуруй и проверь и доработай учебный проект /home/master/AI/projects/boreal-marche-capstone от имени Olga Morgunova
комиты и пуш делать не надо, это будет делать человек после проверки
---
write and correct and save this prompt like files *.md instructions(
example:
PLAN.md
README.md
MASTER_PROMT_*.md
) then run it step by step this corrected prompt(use max detailed like senior prompt engineer from OpenAI, Google, Anthropic for using best practice and full explanation, add all informations that you need for this prompt and ask me some informations if you need it for each step
context:
используй файлы для реализации проекта: ... (repository tree)
презентация - doc/Capstone_Project1_Presentation.html
задание по проекту - doc/Capstone_Project1.pdf
база данных - doc/online_retail_II.xlsx
путь к базе данных - db_path.txt
task:
реализуй первые три этапа проекта из задание по проекту - doc/Capstone_Project1.pdf:
D1 — Project Proposal and Charter
D2 — Model Design and Architecture Report
D3 — Model Implementation and Validation Report
```
