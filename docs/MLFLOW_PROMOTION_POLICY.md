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
   merely as good does not churn the served model. **Data-minimisation clause (D5):** a candidate
   that uses strictly fewer input features may replace the champion if it loses at most 0.005 PR-AUC —
   necessity (Law 25) outranks a marginal metric gain. Used once, for v0.9.5 (37 → 19 features).
5. **Fairness gate (added in D4):** recall of the coupon decision among true lapsers, per region,
   tenure band (<6 m, 6–12 m, >12 m) and spend quartile (edges fixed on the training snapshots;
   groups with ≥ 30 lapsers). The candidate's
   worst gap below overall recall must be ≤ 5 points or, while a known gap is being mitigated (D5),
   no worse than the champion's. Measured on the validation snapshot by
   `boreal_marche.retention.promote`. Known gap at v0.9.5: the top spend quartile is 43 points below
   overall recall on validation (0.46 vs 0.89) — high spenders get high return probabilities, so
   few of the ones who do lapse receive the offer. D5 §7 compares three mitigations; the adopted
   one (a human high-value review list plus a randomised hold-out) does not change the model, so the
   gate stays "no worse than the champion" until the hold-out has measured the coupon's effect.
6. **Tests are green** in CI (unit, contract, golden-prediction regression, data-quality).
7. **Human sign-off** on the pull request opened by the weekly retraining workflow
   (`.github/workflows/retrain.yml`, D4): the workflow
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

## Registry alias versus served artifact (found in D5)

The API in the container serves the committed `artifacts/model.joblib`; the local registry's
`champion` alias had stayed on version 1, an older run, because later retrains did not beat it by
0.005. The weekly CI gate compares against the committed artifact, which is what is served. On
2026-11-01 the alias was moved by hand to version 7 (`0.9.5+xgboost`), with the reason recorded in the
registry tags (`promoted_by=human`) and in `artifacts/model_meta.json → promotion.human_override`.

