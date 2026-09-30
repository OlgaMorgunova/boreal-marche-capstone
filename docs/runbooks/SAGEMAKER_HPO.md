# Runbook — SageMaker hyper-parameter tuning job (D3 "Cloud training and MLOps")

> Alternatives: Azure ML ([AZURE_ML_SWEEP.md](AZURE_ML_SWEEP.md)) or local only — see
> [../CLOUD_OPTIONS.md](../CLOUD_OPTIONS.md). The course specification names SageMaker.

> **Status: PENDING — to be executed by Olga.** Everything below is prepared and tested locally; the
> managed job itself has not been run (the coding agent may not call AWS). Do not report numbers in
> D3 until you have them from the console.

## What is already done (local, reproducible)

| Piece | Where | Verified |
|---|---|---|
| Script-mode entry point (args, `SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`, metric line `cv_pr_auc=`) | `src/boreal_marche/retention/sm_train.py` | `python -m boreal_marche.retention.sm_train` → `cv_pr_auc=0.776837` in ~2 s |
| Self-contained `source_dir` (entry + package copy + pinned requirements) | `boreal_marche.cloud.common.build_source_dir` | ran with `python -I sm_entry.py` (no PYTHONPATH) → `cv_pr_auc=0.778063` |
| Job definition: Bayesian, managed spot, `max_run` 900 s, `max_wait` 1800 s, 20 jobs, 2 parallel, ca-central-1 | `make hpo PROVIDER=aws ARGS=--dry-run` | printed |
| Local Optuna comparison numbers | `reports/metrics/tuning.json` | yes |

## Prerequisites

1. Budget alarm active ([AWS_BUDGET_ALARM.md](AWS_BUDGET_ALARM.md)).
2. An S3 bucket in `ca-central-1`, e.g. `boreal-marche-<initials>`, block public access ON.
3. A SageMaker execution role scoped to **that bucket only** (`s3:GetObject`, `s3:PutObject`,
   `s3:ListBucket` on `arn:aws:s3:::boreal-marche-<initials>[/*]`) + `AmazonSageMakerFullAccess`
   trimmed as far as practical, + CloudWatch Logs write.
4. Short-lived credentials (`aws sso login` or an assumed role). Never put keys in `.env` committed
   files; `.env` is git-ignored and the CI secret scan (gitleaks) runs on every PR.
5. `uv pip install --python .venv/bin/python -e ".[cloud-aws]"` (sagemaker + boto3).
6. Check that the SKLearn framework container you pick has Python ≥ 3.10
   (`--framework-version`, default `1.4-2`); otherwise build a training image from the project.

## Run

```bash
make features                                   # data/gold/retention.parquet must exist
uv pip install --python .venv/bin/python -e ".[cloud-aws]"      # sagemaker + boto3
make hpo PROVIDER=aws ARGS="--bucket boreal-marche-<initials> --role-arn arn:aws:iam::<account>:role/<role> --dry-run"
make hpo PROVIDER=aws ARGS="--bucket boreal-marche-<initials> --role-arn arn:aws:iam::<account>:role/<role>"
# Windows (PowerShell): .venv\Scripts\python scripts\run.py hpo --provider aws --bucket ... --role-arn ...
```

Watch it in SageMaker → Training → Hyperparameter tuning jobs. When it finishes:

## Evidence to capture

The launcher writes `reports/metrics/cloud_hpo_evidence_aws.json` with the job name; complete
it from the console (same fields as below):

```json
{
  "tuning_job_name": "",
  "best_training_job": "",
  "best_cv_pr_auc": 0.0,
  "wall_clock_minutes": 0.0,
  "billable_seconds_total": 0,
  "spot_savings_percent": 0.0,
  "instance_type": "ml.m5.large",
  "cost_usd_from_billing": 0.0,
  "date": "YYYY-MM-DD"
}
```

Screenshots: tuning job summary (best job + objective), training-jobs list with *billable seconds*
and *managed spot savings*, the Billing page a day later with the actual cost.

## Real-time endpoint (deploy once, invoke, tear down)

```python
predictor = tuner.best_estimator().deploy(initial_instance_count=1, instance_type="ml.t2.medium")
# invoke once with a CSV/JSON row, screenshot the response, then IMMEDIATELY:
predictor.delete_endpoint(delete_endpoint_config=True)
```

Teardown checklist (Week 6 checkpoint): endpoint deleted · endpoint config deleted · model deleted ·
no running training jobs · S3 bucket contains only `gold/` and `hpo-output/`.

## Crossover analysis (fill the SageMaker column; local column comes from `tuning.json`)

| | Local Optuna (this laptop, 16 cores) | SageMaker HPO |
|---|---|---|
| Trials | 40 per model (2 models) | 20 |
| Best CV PR-AUC | see D3 table | PENDING |
| Wall-clock | see D3 table | PENDING |
| Cost | $0 (owned hardware) | PENDING |

Managed training becomes worthwhile when one trial no longer fits the laptop's memory or time budget
(e.g. > ~30 min per trial, or > 16 parallel trials needed), when the data cannot leave the cloud
account, or when the job must run unattended on a schedule. At ~2 s per trial on 12k rows, this
project is far below that point: the container start-up (~2–4 min per job) dominates.
