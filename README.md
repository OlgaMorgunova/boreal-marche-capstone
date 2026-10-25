# Boréal Marché — Customer Intelligence Platform

One data pipeline, three model heads — **retention prediction**, **basket recommendations** and a
**grounded bilingual assistant** — for a (fictional) Québec online retailer, built as an end-to-end,
reproducible, governed ML system.

**Student:** Olga Morgunova · **Course:** 420-977-VA AI Development Project (Capstone 1), Vanier
College LEA.EU · **Instructor:** Steve Ataky, Ph.D. · **Individual project**

> **Live:** application and API docs URLs, and the 3–5 min demo video, are added here after the
> first deployment (`docs/runbooks/AWS_APPRUNNER_DEPLOY.md`). Locally: `make up` → http://localhost:8501.

## Results so far (v0.9)

| Head | Metric (held-out, temporal) | Result | Baseline |
|---|---|---|---|
| A — Retention | PR-AUC, test snapshot 2011-09-01 (reported once) | **0.850** [0.835, 0.865] | 0.712 recency deciles · 0.673 "ordered in 60 days" rule · 0.607 dummy |
| A — Retention | Calibration error (ECE) | **0.023** (sigmoid) | 0.053 uncalibrated (validation) |
| A — Decision | Coupon campaign vs. coupon-to-everyone (test quarter, stated assumptions) | **+$55k** per quarter | status quo loses $55k |
| B — Recommend | Hit-rate@5, holdout from 2011-09-01 | **0.145**, 92% of baskets served by rules | 0.017 top-5 best-sellers (8.6×) |
| C — Assistant | Tool choice, 40-question golden set / 20 paraphrases | **97.5% / 95%** local LLM (Ollama llama3.1:8b), 100% out-of-scope refusal | rule baseline 100% / **35%** (keywords overfit the golden set) |
| Platform | API image size · `/predict` p95 in-network | **823 MB** (D3: 1,437 MB) · 13.5 ms | — |
| Monitoring | Shifted cohort (Locust) | 4 of 6 checks red, 7 alert transitions | normal traffic: amber at most |

Honest findings: tuning adds < 0.004 PR-AUC (feature-limited problem); the model is weak for
one-order customers; the profit result is fragile to the assumed save rate; in mlxtend Apriori is
~135× faster than FP-Growth at this size. Details in the [D3 report](reports/977_D3_Morgunova_ImplementationValidation.md).

## Quickstart

**Windows (PowerShell), step by step from an empty machine:** [docs/RUN_FROM_ZERO_WINDOWS.md](docs/RUN_FROM_ZERO_WINDOWS.md).

Linux / macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh              # if uv is not installed
git clone https://github.com/OlgaMorgunova/boreal-marche-capstone.git && cd boreal-marche-capstone
make setup-pinned          # .venv with CPython 3.11 + the exact tested versions (requirements-dev.txt)
make all                   # data -> clean -> features -> dq -> Head A -> Head B  (~12 min)
make test                  # unit, API contract, golden prediction, data quality, pipeline smoke
make reports               # figures + D1-D3 Markdown/HTML/PDF
make api                   # http://127.0.0.1:8000/docs      (or `make up` for the Docker stack)
make frontend              # second terminal: http://127.0.0.1:8501 (prediction, basket, assistant, monitoring, about)
make simulate              # third terminal: Locust traffic, normal then shifted cohort -> Monitoring tab
```

Whole stack in Docker (API, interface, Postgres, Ollama): `docker compose -f docker/compose.yml up --build -d`,
then `python3 scripts/smoke_test.py --api http://127.0.0.1:8000 --ui http://127.0.0.1:8501`.

Without `make` (any OS): `python scripts/run.py all`, `python scripts/run.py test`, … with the venv's
interpreter. Plain pip instead of uv: `python3.11 -m venv .venv && .venv/bin/pip install -r
requirements-dev.txt && .venv/bin/pip install -e . --no-deps`.

| File | Use |
|---|---|
| `requirements.txt` | runtime/serving only (API, assistant, monitoring), pinned, cross-platform |
| `docker/requirements-{api,frontend}.txt` | Linux locks for the two images (`xgboost-cpu`, no training stack) |
| `requirements-dev.txt` | full development machine: runtime + training + interface + Locust + reports + dev tools |
| `pyproject.toml` extras | `train`, `frontend`, `loadtest`, `postgres`, `report`, `notebook`, `dev`, `cloud-aws`, `cloud-azure` |

`make data` downloads the UCI file (or copies `doc/online_retail_II.xlsx` if present) and refuses to
continue if its SHA-256 differs from `params.yaml`. The test set is **not** re-scored on a fresh clone
(test protection); `make evaluate ARGS=--force` re-scores it and records the extra look.
Environment variables: `.env.example`.

**Cloud training backend:** `local` (default, Optuna), `aws` (SageMaker — the course requirement) or
`azure` (Azure ML) — `make hpo PROVIDER=aws ARGS=--dry-run`; see [docs/CLOUD_OPTIONS.md](docs/CLOUD_OPTIONS.md).

## Deliverables

- [x] D1 — Project Proposal and Charter · [`reports/977_D1_Morgunova_ProjectProposal.pdf`](reports/977_D1_Morgunova_ProjectProposal.pdf) (budget-alarm evidence pending)
- [x] D2 — Model Design and Architecture · [`reports/977_D2_Morgunova_DesignArchitecture.pdf`](reports/977_D2_Morgunova_DesignArchitecture.pdf)
- [x] D3 — Model Implementation and Validation · [`reports/977_D3_Morgunova_ImplementationValidation.pdf`](reports/977_D3_Morgunova_ImplementationValidation.pdf) (SageMaker evidence pending)
- [x] D4 — Deployment Demonstration and Dashboard · [`reports/977_D4_Morgunova_DeploymentDemonstration.pdf`](reports/977_D4_Morgunova_DeploymentDemonstration.pdf) (live URLs, CI/CD run and video pending the AWS bootstrap)
- [ ] D5 — Optimization and Ethics
- [ ] D6 — Communication and Presentation
- [ ] D7 — Complete System and Defence

## Repository map

| Path | Contents |
|---|---|
| `params.yaml` | every window, cut-off, cost assumption, grid and seed |
| `src/boreal_marche/` | all business logic (see [ARCHITECTURE.md](ARCHITECTURE.md)) |
| `tests/` | unit, API contract + golden prediction, data quality, end-to-end smoke |
| `data/` | DVC-tracked raw file, generated bronze/silver/gold (not in Git), `manifest.json` |
| `artifacts/` | `model.joblib`, `model_meta.json`, `rules.json`, `index/`, `reference_profile.json`, `demo_customers.json`, `catalogue.json` |
| `reports/` | D1–D3, cleaning report, feature spec, `metrics/*.json`, `figures/` |
| `docs/` | promotion policy, [monitoring policy](docs/MONITORING.md), AWS runbooks, `corpus/` (Head C policies) |
| `docker/`, `.github/workflows/` | two images, compose stack; `ci.yml`, `cd.yml` (OIDC → ECR → App Runner), `retrain.yml`, `monitor.yml` |
| `frontend/`, `loadtest/` | Streamlit interface; Locust traffic simulation |
| `infra/aws/` | IAM trust/permission policies and the one-time bootstrap script |
| `scripts/run.py` | cross-platform task runner (the Makefile delegates to it) |
| `docs/COMMIT_PLAN_TAGS.md`, `docs/COMMIT_PLAN_D4.md` | how the work is split into the `v0.1`…`v0.9` commits and tags |
| `MASTER_PROMPT_D1_D3.md`, `PLAN.md` | the specification this work was executed from, and its progress log |

## Data provenance

UCI Online Retail II (Chen, 2012, DOI 10.24432/C5CG6D, **CC BY 4.0**) — real UK giftware
transactions **re-skinned** as Boréal Marché for a pedagogical scenario; not proprietary client data.
See [DATA_CARD.md](DATA_CARD.md), [MODEL_CARD.md](MODEL_CARD.md), [GOVERNANCE_LOI25.md](GOVERNANCE_LOI25.md).

## What I would do next

Close the fairness gap in the top spend quartile (D5); per-customer expected-value targeting instead
of one threshold; a randomised hold-out to measure coupon uplift; feature ablation for data
minimisation; a durable prediction log (RDS) instead of the per-instance SQLite on App Runner.
