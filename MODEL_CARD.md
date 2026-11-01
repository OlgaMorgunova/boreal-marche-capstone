# Model Card — Head A retention model (`boreal-retention`, v0.9.5+xgboost)

Machine-readable counterpart: [`artifacts/model_meta.json`](artifacts/model_meta.json).
Evidence: [D3 report](reports/977_D3_Morgunova_ImplementationValidation.md) (method, tuning,
calibration, leakage) and [D5 report](reports/977_D5_Morgunova_OptimizationEthics.md)
(minimisation, interpretability, fairness). Status: D5 version.

## Intended use

Estimate, for an **active** customer (≥ 1 order in the last 180 days), the probability of placing at
least one order in the next 90 days, and **propose** whether to include them in the quarterly
retention-coupon campaign (`P(return) ≤ threshold`). A marketing manager approves the list
(Law 25 design consequence — `GOVERNANCE_LOI25.md` §3); the principal factors of each proposal are
available to that reviewer and, on request, to the customer (`POST /predict?explain=3`).

**Out of scope:** credit, pricing or any eligibility decision; new customers without purchase
history; individual-level claims about *why* a person lapses (the factors explain the model, not the
person); measuring coupon uplift (a propensity model trained without randomised treatment data);
unprompted customer-facing explanations in marketing copy; any use on real people — it was trained
on re-skinned public data (`DATA_CARD.md`).

## Model

`RetentionModel` (one joblib file): scikit-learn `Pipeline` [median imputation + missing indicators
→ signed log1p → standardisation; one-hot `season_q`] → XGBoost classifier (Optuna-tuned, 800 trees,
depth 4) → sigmoid (Platt) calibrator fitted on the validation snapshot → threshold **0.61**
(profit-optimal on out-of-fold calibrated validation probabilities).

**19 input features** since v0.9.5 (`features.MODEL_FEATURES`): recency, tenure (and its censoring
flag), orders and revenue over 30/90/180 days, months with an order, order value, unit price, mean
and spread of the gap between orders, overdue ratio, order and revenue trends, calendar quarter.
The data-minimisation audit removed 18 others — time of day, region, cancellations, category mix,
basket shape, outlier share, lifetime value — at a cost of 0.0007 CV PR-AUC (D5 §6).

## Data

Training: 4 quarterly snapshots (cut-offs 2010-06-01, 2010-09-01, 2010-12-01, 2011-03-01; 12,277
rows). Calibration + threshold: 2011-06-01 snapshot (2,658). Test: 2011-09-01 snapshot (2,768),
scored **three times** in total, each after all choices were fixed: D3 (2 looks, explained in D3 §8)
and D5 after the minimisation decision.

## Metrics (test snapshot, 95% bootstrap CI)

| Model | PR-AUC | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| **This model (v0.9.5, 19 features)** | **0.847** [0.832, 0.863] | 0.767 | 0.192 | 0.041 |
| Previous model (v0.3/v0.9, 37 features) | 0.850 [0.835, 0.865] | 0.772 | 0.189 | 0.023 |
| Recency-decile heuristic | 0.712 | — | — | — |
| Rule "ordered in last 60 days" | 0.673 | — | — | — |
| Dummy (prior) | 0.607 | 0.500 | — | — |

Validation PR-AUC 0.805 (previous 0.809). Campaign under the stated assumptions (D3 §7, save rate
20%): on the test quarter the proposal list loses $1,039 against contacting nobody and is **$54,137
better than the current coupon-to-everyone policy**. The profit curve is flat near its optimum: the
threshold moved from 0.53 to 0.61 between retrains, and the result is sensitive to the save rate (at
10% no campaign is profitable). D3's test policy figures were computed with threshold 0.56 from an
earlier training run; the threshold above is the one served.

## Subgroup performance (fairness audit, D5 §7)

Lapser recall = share of customers who lapsed who were proposed for the offer.

| Group | Validation lapser recall | Test lapser recall | Flag |
|---|---:|---:|---|
| Overall | 0.89 | 0.68 | — |
| Spend Q1 (low) | 1.00 | 0.94 | — |
| Spend Q4 (top) | 0.46 | 0.16 | **flagged** |
| Tenure > 12 months | 0.83 | 0.57 | **flagged** |
| Region ON | 0.92 | 0.60 | flagged on test only |

The gap is caused by the single value-blind threshold, not by worse ranking in those groups (top
spend-quartile PR-AUC 0.96 on test). Mitigation adopted: human high-value review list + randomised
hold-out before changing the decision rule (`GOVERNANCE_LOI25.md` §7) — **built and served in
v1.0**: `/predict` returns an `action` (coupon / holdout / review / none) and `GET /campaign` the
list for approval. If the reviewer approves the whole list, top-quartile lapser recall rises to
0.89 (validation) and 0.82 (test); this measures the list, not reviewers' real choices.

## Interpretability

Exact per-prediction factors: XGBoost TreeSHAP contributions scaled by the Platt slope, summing to
the calibrated log-odds (max error 4e-06), folded to input features; `season_q` is treated as the
baseline. Most influential globally: months with an order in the last 180 days, mean gap between
orders, 180-day revenue, gap regularity, tenure. Top-3 factor set stable under ±5% input noise for
70% of customers. Correlated features share credit, so a near-threshold customer's "reason" is
fragile — see D5 §5 for the assessment of customer-facing use.

## Limitations and known failure modes

- Weak for one-order and low-spend customers (test PR-AUC 0.52–0.59): little history to learn from.
- High-spend and long-tenure lapsers are rarely proposed (above).
- Seasonal base-rate shift (38%–62% across quarters); monitoring and quarterly recalibration needed.
- Regions are a synthetic mapping; subgroup results by region describe the mapping, not Québec.
- Serialised with joblib/pickle: load only from the project registry/repository; pinned library
  versions are recorded in the sidecar.

## Ethical considerations

Profiling of purchase behaviour → disclosed in the bilingual notice; human approval of every
campaign; 18 features removed as unnecessary and no longer stored; no personal identifiers are model
inputs; region kept out of the model and used only to audit fairness.
