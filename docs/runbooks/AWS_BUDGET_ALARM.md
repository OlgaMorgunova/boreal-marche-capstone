# Runbook — AWS budget alarm ($10) · evidence for D1

> **Status: PENDING — to be executed by Olga.** The coding agent is not permitted to run the AWS CLI.
> Required by assignment §8.4 ("A budget alarm at $10 must be configured in Week 1 and evidenced in
> Deliverable 1").

## Steps (AWS Console, ~5 minutes)

1. Sign in as the account root **once** only to enable *IAM user/role access to Billing*
   (Account → IAM user and role access to Billing information → Activate). Then sign out of root.
2. Billing and Cost Management → **Budgets** → *Create budget* → *Use a template* →
   **Monthly cost budget**.
3. Name `boreal-marche-capstone`, amount **10.00 USD**, e-mail recipient = your address.
   The template creates alerts at 85% actual and 100% forecasted — add one more at **50% actual**.
4. Create. Confirm the subscription e-mail if prompted.
5. (Recommended) Cost Explorer → enable; Billing preferences → enable *Free Tier usage alerts*.

## Evidence to capture for the D1 PDF

- Screenshot of the budget list showing `boreal-marche-capstone`, 10.00 USD, alert thresholds.
- Screenshot of the alert configuration (recipients visible, e-mail partially masked).
- Date configured: `____-__-__` (fill in).

Insert both into `reports/D1_Project_Proposal_Charter.md` §9 where marked `PENDING`, then
`make reports`.

## Cost ceiling for the project

| Item | Plan | Ceiling |
|---|---|---|
| SageMaker HPO (20 trials, ml.m5.large, managed spot, 15 min cap) | ≈ 20 × 3 min × $0.12/h × ~0.3 spot ≈ **$0.05–0.50** | $2 |
| Real-time endpoint (deployed once, invoked, torn down ≤ 1 h) | ml.t2.medium ≈ $0.06/h | $1 |
| S3 (≤ 1 GB, a few months) | < $0.10 | $0.50 |
| ECR + App Runner / Render (D4) | App Runner min. ≈ $5/month if left on; Render free tier = $0 | $8 |
| Hosted LLM (Head C evaluation, synthetic questions only) | < $1 | $2 |
| **Total** | **target < $12** (excellent band) | **hard stop $25** |

Prices are indicative (ca-central-1, 2026) and must be re-checked in the AWS pricing pages before
running; record the *actual* billed amount in D3/D5, not this estimate.
