# Architecture

![Architecture](reports/figures/architecture.png)

**Principle:** mining and training are expensive and run **offline**; serving is a lookup and runs
**online**. Only artifacts cross the line — never the algorithm. Full walkthrough and rationale:
[D2 report](reports/977_D2_Morgunova_DesignArchitecture.md).

## Offline (Make targets, scheduled retraining in D4)

| Stage | Command | Output |
|---|---|---|
| Ingest (bronze) | `make data` | `data/bronze/transactions.parquet`, `data/manifest.json` |
| Clean (silver) | `make etl` | `data/silver/lines.parquet`, `cancellations.parquet`, `reports/cleaning_report.md` |
| Features (gold) | `make features` | `data/gold/retention.parquet`, `reports/feature_spec.md` |
| Data quality | `make dq` | pass/fail per layer |
| Head A | `make spotcheck tune train evaluate leakage` | `artifacts/model.joblib`, `model_meta.json`, MLflow runs + registry |
| Head B | `make rules-sweep rules rules-eval` | `artifacts/rules.json` |
| Head C (Week 9) | `make index` | `artifacts/index/` |
| Reports | `make reports` | `reports/977_D*.{md,html,pdf}`, `reports/figures/` |

## Online (FastAPI, `src/boreal_marche/api.py`)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness — the process is up |
| `GET /ready` | readiness — model and rules loaded (503 otherwise) |
| `GET /version` | API version, git commit, model version and source, rules date and count |
| `POST /predict`, `/predict/batch` | calibrated P(return in 90 d) + proposed coupon action (human approves) |
| `POST /recommend` | top-k items, firing rule per item, strategy (`rules`, `rules+popularity`, `popularity_fallback`, `empty_basket`) |
| `POST /chat`, `GET /metrics` | D4 (Head C, monitoring) |

Artifacts are loaded once in the lifespan hook: the model from `models:/boreal-retention@champion`
with fallback to `artifacts/model.joblib`; rules from `artifacts/rules.json`. Request and response
models are Pydantic v2 with `extra="forbid"`.

## Code map

```
src/boreal_marche/
  config.py            typed params.yaml + paths
  data/ingest.py       bronze + manifest + checksum
  data/cleaning.py     silver rules + cleaning report
  data/quality.py      data-quality assertion suite
  reskin.py            documented re-skin mapping
  features.py          point-in-time gold snapshots + FEATURE_SPEC
  retention/           models, cv, spotcheck, tune, train, artifact, profit, evaluate, leakage, sm_train
  recommend/           baskets, mine, engine, evaluate
  registry.py          champion alias -> local fallback
  schemas.py, api.py   typed API
  reporting/           figures and report builder
```
