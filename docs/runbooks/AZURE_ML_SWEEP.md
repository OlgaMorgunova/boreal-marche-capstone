# Runbook — Azure ML sweep job (alternative to SageMaker)

> **Status: PENDING — optional, to be executed by Olga after instructor approval** (the course
> specification names SageMaker; see [../CLOUD_OPTIONS.md](../CLOUD_OPTIONS.md)). Prepared and
> dry-run tested locally; never run by the coding assistant.

## 1. Budget first ($10) — Cost Management

1. Portal → **Cost Management + Billing** → select the subscription → **Budgets** → *Add*.
2. Name `boreal-marche-capstone`, reset period *Monthly*, amount **10** (subscription currency).
3. Alert conditions: *Actual* 50%, *Actual* 85%, *Forecasted* 100%; recipient = your e-mail. Create.
4. Screenshot the budget list and the alert conditions (this replaces the AWS evidence in D1 §9
   only if the instructor approved Azure).

Budgets alert; they do not stop spending — the teardown in §5 is what guarantees the cost stays low.

## 2. Workspace (once, ~10 minutes)

```bash
az login
az group create --name rg-boreal-marche --location canadacentral
az extension add --name ml          # Azure ML CLI v2 (optional; the SDK below is enough)
az ml workspace create --name mlw-boreal-marche --resource-group rg-boreal-marche --location canadacentral
```

Or in the portal: *Azure Machine Learning* → *Create*, region **Canada Central**. The workspace also
creates a storage account, a key vault and Application Insights; the first custom environment build
creates a **container registry (billed daily)** — another reason to delete the resource group at
the end.

## 3. Install and dry-run

```bash
uv pip install --python .venv/bin/python -e ".[cloud-azure]"     # azure-ai-ml + azure-identity
make features                                                     # data/gold/retention.parquet
make hpo PROVIDER=azure ARGS="--subscription <sub-id> --resource-group rg-boreal-marche --workspace mlw-boreal-marche --dry-run"
# Windows (PowerShell):
# .venv\Scripts\python scripts\run.py hpo --provider azure --subscription <sub-id> `
#     --resource-group rg-boreal-marche --workspace mlw-boreal-marche --dry-run
```

Check the printed definition: compute `boreal-cpu`, `Standard_DS3_v2`, tier `low_priority`,
`min_instances 0`, `max_instances 2`; sampling `bayesian`; primary metric `cv_pr_auc`; limits
20 trials, 2 concurrent, 1,800 s timeout; the command passes `--log-mlflow` (Azure sweeps read the
metric from MLflow).

## 4. Run

Same command without `--dry-run`. The launcher creates/updates the compute cluster, uploads
`data/gold/`, builds the environment (Python 3.11 + pinned requirements + `azureml-mlflow`), submits
the sweep and writes `reports/metrics/cloud_hpo_evidence_azure.json` with the job name. Follow it in
Azure ML studio → *Jobs* → `boreal-retention-hpo`.

Evidence: sweep overview (best trial + `cv_pr_auc`), trials table, compute page showing
*Low priority*, and Cost Management → *Cost analysis* filtered to the resource group (a day later).

**Real-time endpoint equivalent (only if required):** deploy the best model once as a *managed online
endpoint* on the smallest SKU, invoke it once, screenshot, then
`az ml online-endpoint delete --name <name> --yes` immediately.

## 5. Teardown checklist

- [ ] no running jobs · [ ] compute cluster at 0 nodes (or deleted:
  `az ml compute delete --name boreal-cpu -g rg-boreal-marche -w mlw-boreal-marche --yes`)
- [ ] online endpoint deleted (if created)
- [ ] after the evidence is saved: `az group delete --name rg-boreal-marche --yes` (removes
  storage, registry, key vault — everything billable)

## 6. Local-vs-managed comparison

Fill the Azure column of the D3 §11 table (best CV PR-AUC, wall-clock, cost) exactly as for SageMaker;
the crossover analysis there applies unchanged (container start-up dominates at ~2 s per trial).
