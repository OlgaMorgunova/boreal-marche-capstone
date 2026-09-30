# MLflow model promotion policy — Head A (retention)

**Owner:** Olga Morgunova · **Registry model:** `boreal-retention` · **Applies from:** v0.3

## Aliases

| Alias | Meaning | Who moves it |
|---|---|---|
| `champion` | The version the API serves (`models:/boreal-retention@champion`) | `make train` if every gate passes; otherwise a human |
| `challenger` | Newest candidate that did **not** pass, or the runner-up family | `make train` |

Stages (`Staging`/`Production`) are deprecated in MLflow 2.x; aliases replace them.

## Promotion gates (evaluated mechanically in `retention/train.py::promotion_decision`)

A candidate becomes `champion` only if **all** hold on the **validation snapshot** (never the test):

1. **Beats the best baseline** (dummy prior, recency-60 rule, recency deciles) by **≥ 0.05 PR-AUC**.
2. **Calibration helps or is neutral:** calibrated Brier ≤ raw Brier.
3. **Calibrated:** expected calibration error (ECE, 10 bins) **≤ 0.05**.
4. **Beats the current champion** by **≥ 0.005 PR-AUC** (or no champion exists). A retrain that is
   merely as good does not churn the served model.
5. **Tests are green** in CI (unit, contract, golden-prediction regression, data-quality).
6. **Human sign-off** on the pull request opened by the weekly retraining workflow (D4): the workflow
   never deploys; a person merges. This is also the Law 25 human-review step for the model itself.

The decision, each check's result and the lineage (git SHA, raw-data SHA-256, gold SHA-256, run id)
are written to `artifacts/model_meta.json → promotion` and to the MLflow run tags.

## Rollback

Move the `champion` alias back to the previous version (`MlflowClient().set_registered_model_alias`)
and restart the service. No rebuild, no redeploy. If the registry is unreachable the API serves the
committed `artifacts/model.joblib` and `/version` reports `model_source`.

## What a "model version" contains

Preprocessing + estimator + calibrator + decision threshold in **one** joblib artifact, the metadata
sidecar (feature order, library versions, training cut-offs, row counts, metrics, git commit, MLflow
run id), and the MLflow run with parameters, metrics and plots. The test snapshot metric is added to
the sidecar exactly once by `make evaluate`.
