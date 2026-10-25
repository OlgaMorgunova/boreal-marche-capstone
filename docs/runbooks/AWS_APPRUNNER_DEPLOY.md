# Runbook — deploy to AWS App Runner with GitHub OIDC (D4)

Result: `https://<id>.us-east-1.awsapprunner.com` (interface) and `https://<id>…/docs` (API
documentation), redeployed automatically by `.github/workflows/cd.yml` on every merge to `master`,
with SHA-tagged images in ECR and no AWS key stored in GitHub.

```
merge to master ─► GitHub Actions (cd.yml) ──OIDC token──► STS AssumeRoleWithWebIdentity
                     │                                       (role BorealGithubDeploy, 1 h, trusts
                     │                                        repo:…:environment:production only)
                     ├─ docker build ─► ECR boreal-api:<sha>, boreal-ui:<sha>   (immutable tags)
                     ├─ apprunner update-service ─► boreal-api (1 vCPU/2 GB), boreal-ui (0.25/0.5)
                     └─ scripts/smoke_test.py ─► /version == <sha>, golden prediction, chat, UI health
```

## 0. Before you start (10 min)

1. **Admin SSO profile, not root** — done 2026-10-24: IAM Identity Center user `admin_boreal`,
   CLI profile `admin-profile` (`AdministratorAccess`). The `[default]` profile on this PC is still the
   root login: always pass `--profile admin-profile` (the bootstrap script sets it itself). If you
   created root access keys earlier, delete them (IAM → Security credentials). Use the same profile
   for SageMaker: `$env:AWS_PROFILE="admin-profile"` before `scripts/run.py hpo --provider aws …`.
   Each morning: `aws sso login --profile admin-profile` (the session expires).
2. **Region: us-east-1 for serving.** App Runner is not offered in `ca-central-1` (console, checked
   2026-10-24: "AWS App Runner is not available in Canada (Central)"). The serving stack (ECR + App
   Runner) runs in `us-east-1` (N. Virginia, the supported region closest to Montréal); SageMaker HPO
   stays in `ca-central-1` as the course requires. Why this is acceptable here: the deployed demo
   processes only the public, pseudonymised UCI data (no real personal information), and no
   customer data is stored outside the image except the ephemeral request log. For real Boréal Marché
   customers this would be a Law 25 transfer outside Québec (s. 17 assessment) and the design would
   change to ECS Fargate + ALB in `ca-central-1` (≈ $30/month more) — recorded in D4 §6.
3. The $10 budget alarm from D1 stays on — budgets are account-wide, so it covers us-east-1 too.

## 1. One-time bootstrap (≈ 20 min, mostly waiting)

From the repository root, Docker Desktop running, signed in with the admin SSO profile:

```powershell
aws sso login --profile admin-profile
aws sts get-caller-identity --profile admin-profile     # must show AWSReservedSSO_AdministratorAccess
powershell -ExecutionPolicy Bypass -File infra\aws\bootstrap.ps1 -AwsProfile admin-profile
# defaults: -Region us-east-1 -Repo OlgaMorgunova/boreal-marche-capstone
```

It creates the OIDC provider, two ECR repositories (scan on push, immutable tags, keep last 10), the
App Runner ECR-access role, the deploy role `BorealGithubDeploy` (trust:
`infra/aws/github-oidc-trust.json`, permissions: `infra/aws/github-deploy-policy.json` — push to the
two repositories, update/describe the two services, pass the ECR-access role to App Runner, nothing
else), pushes a first image of each service and creates the two services (single instance, health
checks on `/ready` and `/_stcore/health`). It ends by printing seven values.

## 2. GitHub settings (5 min)

- *Settings → Secrets and variables → Actions → **Variables***: add `AWS_REGION` (`us-east-1`),
  `AWS_DEPLOY_ROLE_ARN`, `APPRUNNER_ECR_ROLE_ARN`, `APPRUNNER_API_ARN`, `APPRUNNER_UI_ARN`, `API_URL`, `UI_URL` (values
  printed by the script). They are identifiers, not credentials — there are **no secrets** to add.
- *Settings → Environments → New environment* `production` → *Deployment branches*: `master` only.
  The trust policy only accepts tokens whose subject is `repo:OlgaMorgunova/boreal-marche-capstone:environment:production`.
- *Settings → Branches → Add rule* for `master`: require a pull request, require status checks
  `lint · types · tests · data quality · assistant eval`, `docker build · size · container smoke test`
  and `secret scan`, do not allow bypassing.

## 3. First automated deployment (evidence for D4)

1. Open a PR from `feat/d4` → CI runs (screenshot the green checks: *CI evidence*).
2. Merge it (merge commit) → *Actions → CD* runs. Screenshot/download:
   - the step *Show the assumed identity* (an `assumed-role/BorealGithubDeploy/gh-cd-…` ARN: *OIDC evidence*);
   - the image list with the SHA tag (job summary) and ECR → `boreal-api` → image tag = commit SHA;
   - the *Post-deploy smoke test* output (`version matches commit … PASS`).
3. Check by hand: `curl $API_URL/version` → `git_commit` equals `git rev-parse origin/master`.
4. Populate the dashboard on the live service (shifted cohort included):
   `python scripts/run.py simulate --host $API_URL --users 5 --seconds 120`, then open the
   *Monitoring* tab: red checks and the alert history are the D4 evidence.

## 4. Rollback, pause, cost

- **Rollback** = redeploy the previous image (tags are immutable, so it is exactly what ran before):
  `python scripts/apprunner_deploy.py --service-arn <api arn> --image <registry>/boreal-api:<previous sha> --port 8000 --access-role-arn <ecr role> --env GIT_SHA=<previous sha> --env IMAGE_TAG=<previous sha>`
  (needs admin credentials locally; the deploy role is only assumable from GitHub).
- **Pause when not grading** (no compute charge while paused, URL kept):
  `aws apprunner pause-service --service-arn <arn>` / `resume-service`. Resume the morning of grading
  and run the smoke test (§8.2 of the course: a URL down at grading time counts as not submitted).
- **Cost** (us-east-1 list prices, verify on the pricing page): provisioned memory ≈ $0.007/GB-hour,
  active vCPU ≈ $0.064/vCPU-hour. API 2 GB ≈ $10/month idle + pennies of active CPU; UI 0.5 GB ≈
  $2.6/month; ECR storage < $0.10. ≈ **$13/month while running**, $0 while paused. Keeping both
  services up from week 12 to week 15 (≈ 1 month) stays inside the $25 project ceiling together with
  the SageMaker HPO job; pausing between grading windows keeps it near $5.
- **Teardown after D7:** delete both services, the auto-scaling configuration, the ECR repositories,
  the two roles and the OIDC provider.

## 5. What the deployed configuration means

- *Single instance* (`max-size 1`): the SQLite prediction log lives on the instance; two instances
  would split the monitoring data. The log is lost on redeploy — acceptable for a demo, documented in
  D4; the Compose stack uses Postgres, which is the production design (RDS) at higher cost.
- *Ollama is not deployed* (an 8B model needs a GPU instance, > $300/month). The live assistant offers
  the rule provider; the Ollama provider runs in the Compose stack and in the evaluation.
- The MLflow registry is not reachable from App Runner, so the image serves the committed artifact
  (`BOREAL_MODEL_SOURCE=local`) and `/version` says so.

## 6. Fallback: Render (only if App Runner is unavailable)

Keep CI and the OIDC→ECR push as they are (OIDC evidence unchanged). Push the interface and API images
also to GHCR (public), create two Render *Web Services* from those images, and replace the two
`apprunner_deploy.py` steps with a `curl -X POST <render deploy hook>?imgURL=ghcr.io/…:<sha>` (the hook
URL is a secret). Free instances sleep after 15 minutes: open the URLs a minute before grading.
