# Monitoring policy — Boréal Marché platform

**Owner:** Olga Morgunova · **Applies from:** v0.9 (D4) · Thresholds live in `params.yaml → monitoring`
and are the single source of truth; this page explains them.

## What is logged

Every request is timed by the API middleware and written to `request_log` (route template, status,
latency, caller, request id) and as a structured JSON line on stdout. Every prediction is written to
`prediction_log` with its full feature vector, probability, decision, threshold and model version;
every recommendation to `recommendation_log` (basket size, unknown SKUs, strategy that fired); every
assistant turn to `chat_log` (provider, tool, refusal, handoff, latency). Storage: SQLite on the App
Runner instance (single instance by design), Postgres in the Compose stack. Retention: 13 months
(`LogStore.purge`, privacy notice).

## Reference

The calibration snapshot (2011-06-01) minus the 300 held-out demo customers — the most recent snapshot
of the model's training pipeline. The pooled training snapshots were rejected as the reference after
measurement: healthy next-quarter traffic was red on lifetime features (`tenure_days` PSI 1.15,
`orders_lifetime`, `revenue_lifetime`) and on cadence missingness, because the dataset starts in
December 2009 and every later snapshot has a longer observable history. That known shift is stored in
`artifacts/reference_profile.json → known_shift_vs_pooled_training`. `season_q` is excluded: it is
constant within a quarter by construction.

## Checks and thresholds

Computed on the last 500 predictions / 2,000 requests (`GET /monitoring/report`); fewer than 100
predictions → *insufficient data* (grey), never green.

| Check | Measure | Amber | Red | What it triggers |
|---|---|---|---|---|
| Input drift | PSI per feature (decile bins + missing bin) and KS statistic vs reference | any feature PSI ≥ 0.10 or KS ≥ 0.10 | ≥ 10% of features with PSI ≥ 0.25 | Investigate the top drifted features; if a population change, schedule retraining |
| Prediction drift | PSI of `p_return` vs reference predictions; mean p; proposed-coupon rate | PSI ≥ 0.10 | PSI ≥ 0.25 | Freeze the offer list (human review is already mandatory); check upstream features |
| Data quality | max change in a feature's missing rate; share of `/predict` rejected (422) | ≥ 0.05 | ≥ 0.15 | Data-pipeline incident: check the feature producer before trusting scores |
| Error rate | share of 5xx responses | ≥ 1% | ≥ 5% | Operational incident; roll back the image (previous SHA tag) if a deploy caused it |
| Latency | server-side p95 of `/predict` | ≥ 150 ms | ≥ 500 ms | Scale up the instance / profile (D5) |
| Rule fire-rate | share of non-empty baskets answered by a rule (lower is worse) | < 0.50 | < 0.30 | New catalogue items not covered by rules: re-mine (`make rules`) |

## Alerting loop

1. The API re-evaluates every 60 s in a background thread; a check that **changes level** writes an
   `alert_log` row, a structured `alert` log line (CloudWatch Logs on App Runner) and, if
   `ALERT_WEBHOOK_URL` is set, posts to a Slack/Teams/Discord webhook. A sustained breach notifies
   once; recovery to green is also notified.
2. `.github/workflows/monitor.yml` polls the deployed `/monitoring/report` hourly and opens (or comments
   on) a `monitoring-alert` GitHub issue when the service is red or unreachable — this path still works
   when the service itself is down.

## Delayed ground truth

The label (an order within 90 days) exists only 90 days after a prediction. Backfill design
(`boreal_marche.monitoring.backfill`): a weekly job joins predictions older than 90 days with the
customer's orders in `(prediction time, +90 days]` and writes `ground_truth`; the report then shows
realised PR-AUC and lapser recall. **The first true accuracy signal arrives 90 days after the first
logged prediction, plus up to 7 days of job cadence** (the dashboard prints the date). Until then the
leading indicators are input drift, prediction drift and data quality. In this repository the demo
customers' outcomes are known historically, so `make backfill` replays them to prove the pipeline;
those labels are in-sample for the calibrator and are not a new performance estimate.

## Retraining decision rule

Retrain (the weekly workflow does it automatically and opens a PR only if the promotion gates pass) when
any of: input drift red for 7 consecutive days; prediction drift red for 3 days; realised PR-AUC (once
available) more than 0.05 below the validation PR-AUC (0.809); a new quarter starts (`season_q`
changes). Promotion still requires every gate in `docs/MLFLOW_PROMOTION_POLICY.md`, including the
fairness gate, and a human merge.
