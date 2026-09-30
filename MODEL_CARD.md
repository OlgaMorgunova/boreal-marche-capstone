# Model Card — Head A retention model (`boreal-retention`, v0.3.0+xgboost)

Machine-readable counterpart: [`artifacts/model_meta.json`](artifacts/model_meta.json).
Evidence: [D3 report](reports/977_D3_Morgunova_ImplementationValidation.md). Status: D3 version;
fairness audit and interpretability are completed in D5.

## Intended use

Estimate, for an **active** customer (≥ 1 order in the last 180 days), the probability of placing at
least one order in the next 90 days, and **propose** whether to include them in the quarterly
retention-coupon campaign (`P(return) ≤ threshold`). A marketing manager approves the list
(Law 25 design consequence — `GOVERNANCE_LOI25.md`).

**Out of scope:** credit, pricing or any eligibility decision; new customers without purchase
history; individual-level claims about *why* a person lapses; measuring coupon uplift (the model is a
propensity model, trained without randomised treatment data); any use on real people — it was
trained on re-skinned public data (`DATA_CARD.md`).

## Model

`RetentionModel` (one joblib file): scikit-learn `Pipeline` [median imputation + missing indicators
→ signed log1p → standardisation; one-hot `region`, `season_q`] → XGBoost classifier (Optuna-tuned)
→ sigmoid (Platt) calibrator fitted on the validation snapshot → threshold 0.56. 37 point-in-time
features (`reports/feature_spec.md`).

## Data

Training: 4 quarterly snapshots (cut-offs 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01; 12,277
rows). Calibration + threshold: 2011-06-01 snapshot (2,658). Test: 2011-09-01 snapshot (2,768),
evaluated after all choices were fixed (2 looks, explained in D3 §8).

## Metrics (test snapshot, 95% bootstrap CI)

| Model | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|
| This model (calibrated) | **0.850** [0.835, 0.865] | 0.772 | 0.189 |
| Recency-decile heuristic | 0.712 | — | — |
| Rule "ordered in last 60 days" | 0.673 | — | — |
| Dummy (prior) | 0.607 | 0.500 | — |

ECE 0.023 on test. Campaign under stated assumptions (D3 §7): the chosen threshold is ≈ break-even
on the test quarter and **$55k better than coupon-to-everyone**; results are sensitive to the
assumed 20% save rate (at 10% no campaign is profitable).

## Limitations and known failure modes

- Weak for one-order and low-spend customers (PR-AUC ≈ 0.54–0.61): little history to learn from.
- A single probability threshold rarely selects high-spend lapsers (lapser recall 0.18 in the top
  spend quartile vs. 0.88 in the lowest) — a value-blind decision; D5 evaluates the per-customer
  expected-value policy and the fairness implications.
- Seasonal base-rate shift (38%–62% across quarters); monitoring and quarterly recalibration needed.
- Regions are a synthetic mapping; subgroup results by region describe the mapping, not Québec.
- Serialised with joblib/pickle: load only from the project registry/repository; pinned library
  versions are recorded in the sidecar.

## Ethical considerations

Profiling of purchase behaviour → disclosed in the bilingual notice; human approval of every campaign;
features audited for necessity (time-of-day features are removal candidates); no personal
identifiers are model inputs.
