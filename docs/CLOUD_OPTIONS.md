# Cloud options — local, AWS or Azure

The managed hyper-parameter search (D3 §11) can run on three interchangeable backends. All three
execute the **same** entry point (`boreal_marche.retention.sm_train`) on the **same** gold table, so
their best scores are directly comparable with the local Optuna run.

> **Course compliance.** The specification (§3.5, §3.6, §8.4) names **Amazon SageMaker**, an AWS
> budget alarm and AWS OIDC for CD. `aws` is therefore the default for grading. `azure` is a
> technically equivalent path; use it **only with the instructor's written approval** (ask before
> Week 6), and keep the local comparison either way. `local` alone does not satisfy the SageMaker
> evidence criterion — report it honestly as a gap if cloud access is impossible.

## Choosing the backend

| | `local` | `aws` (default for grading) | `azure` |
|---|---|---|---|
| Service | Optuna TPE + median pruning (`retention/tune.py`) | SageMaker HyperparameterTuner, Bayesian | Azure ML sweep job, Bayesian sampling |
| Compute | this machine | `ml.m5.large`, **managed spot** | `Standard_DS3_v2`, **low-priority** (spot equivalent), cluster scales 0 → 2 |
| Region | — | `ca-central-1` (Montréal) | `canadacentral` (Toronto) |
| Auth | none | `aws sso login` / assumed role; execution role scoped to one bucket | `az login` (`DefaultAzureCredential`); role *AzureML Data Scientist* on the workspace |
| Extra install | none | `pip install -e ".[cloud-aws]"` | `pip install -e ".[cloud-azure]"` |
| Cost guard | $0 | $10 AWS Budget ([runbook](runbooks/AWS_BUDGET_ALARM.md)), `max_run`/`max_wait` caps | $10 Cost Management budget ([runbook](runbooks/AZURE_ML_SWEEP.md)), sweep `timeout`, min 0 nodes |
| Indicative cost of 20 trials | $0 | ≈ $0.05–0.50 | ≈ $0.05–0.50 compute + registry/storage while the workspace exists |
| Free credits | — | Free Tier does not cover SageMaker training | *Azure for Students*: US$100 credit, no card |
| Search-space fidelity | full (log priors, pruning) | full (log scaling) | log ranges become an 8-value log-spaced `choice` (Bayesian sampler supports only choice/uniform/quniform); no early termination with Bayesian |
| CD with OIDC (D4) | — | GitHub OIDC → IAM role (`aws-actions/configure-aws-credentials`) | GitHub OIDC → federated credential (`azure/login@v2`) |
| Serving (D4) | Docker locally | App Runner (or Render, provider-neutral) | Azure Container Apps (or Render) |

## Switching

```bash
# one-off, any OS
python -m boreal_marche.cloud --provider local|aws|azure [--dry-run] [provider arguments]
# via the task runner / make
python scripts/run.py hpo --provider azure --dry-run
make hpo PROVIDER=aws ARGS="--bucket B --role-arn ARN"
# default for the project: params.yaml -> cloud.provider (or HPO_PROVIDER in .env)
```

`--dry-run` prints the full job definition (instance, caps, search space, command) and builds the
self-contained source directory without touching any cloud account — CI runs exactly this.

Provider arguments can come from the environment (`.env.example`): `SAGEMAKER_BUCKET`,
`SAGEMAKER_ROLE_ARN` for AWS; `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZUREML_WORKSPACE`
for Azure.

## After a real run

The launcher writes `reports/metrics/cloud_hpo_evidence_<provider>.json` with the job name; fill in
best score, wall-clock, billable seconds, spot/low-priority savings and the billed cost, take the
screenshots listed in the provider runbook, then update D3 §11 and run `make reports`.

## Data protection note (Law 25, s. 17)

Only the de-identified gold feature table (opaque customer ids, no names/e-mails) is uploaded, to a
Canadian region. Both providers encrypt at rest by default. Delete the bucket/resource group after
the evidence is captured (retention policy, `GOVERNANCE_LOI25.md` §6).
