<#
One-time AWS setup for D4 continuous deployment (run by a human, once, from the repository root):

    aws sso login --profile admin-profile
    powershell -ExecutionPolicy Bypass -File infra\aws\bootstrap.ps1 -AwsProfile admin-profile

Creates, in us-east-1 (App Runner is not offered in ca-central-1; SageMaker stays in ca-central-1):
  1. the GitHub OIDC identity provider (if missing);
  2. ECR repository boreal-api (scan on push, immutable tags, keep last 10 images);
  3. role BorealAppRunnerEcrAccess (App Runner pulls private ECR images);
  4. role BorealGithubDeploy, trusted ONLY by this repository's "production" environment via OIDC,
     with a least-privilege inline policy (push to boreal-api, update the boreal-api service);
  5. a first API image (built and pushed from this machine) and the App Runner service boreal-api
     (single instance: the SQLite prediction log lives on the instance).
The Streamlit interface is not on App Runner (its proxy rejects WebSockets): it runs on Streamlit
Community Cloud (runbook section 2b).
Then prints the GitHub repository variables to set. Idempotent where AWS allows it.

Prerequisites: AWS CLI v2 signed in with an administrator identity (an IAM Identity Center user or
IAM admin user — NOT root access keys), Docker running. Step-by-step: docs/runbooks/AWS_APPRUNNER_DEPLOY.md
#>
param(
    [string]$AwsProfile = "admin-profile",
    [string]$Region = "us-east-1",
    # Exactly as GitHub writes it in the OIDC "sub" claim: owner and repository carry their immutable
    # numeric IDs (seen in CloudTrail, 2026-10-25). Find yours in the AccessDenied event's userName.
    [string]$Repo = "OlgaMorgunova@169389466/boreal-marche-capstone@1370564309",
    [string]$ApiCpu = "1 vCPU",
    [string]$ApiMemory = "2 GB"
)
$ErrorActionPreference = "Stop"
$env:AWS_PROFILE = $AwsProfile       # never the [default] profile (root login on this PC)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmp = New-Item -ItemType Directory -Force (Join-Path ([IO.Path]::GetTempPath()) "boreal-aws")

function Render([string]$name) {
    $text = (Get-Content (Join-Path $here $name) -Raw).Replace("__ACCOUNT_ID__", $Account).Replace("__REGION__", $Region).Replace("__REPO__", $Repo)
    $out = Join-Path $tmp $name
    Set-Content -Path $out -Value $text -Encoding ascii
    return "file://$($out -replace '\\','/')"
}
function Exists([scriptblock]$probe) { try { & $probe 2>$null | Out-Null; return $LASTEXITCODE -eq 0 } catch { return $false } }

$Account = aws sts get-caller-identity --query Account --output text
$Caller = aws sts get-caller-identity --query Arn --output text
if ($Caller -like "*:root") { Write-Warning "You are signed in as the ROOT user. Create an admin user in IAM Identity Center and use it instead (aws configure sso)." }
Write-Host "Account $Account, region $Region, repository $Repo"

# 1. OIDC provider --------------------------------------------------------------------------------
$oidcArn = "arn:aws:iam::${Account}:oidc-provider/token.actions.githubusercontent.com"
if (-not (Exists { aws iam get-open-id-connect-provider --open-id-connect-provider-arn $oidcArn })) {
    aws iam create-open-id-connect-provider --url https://token.actions.githubusercontent.com `
        --client-id-list sts.amazonaws.com | Out-Null
    Write-Host "created OIDC provider"
}

# 2. ECR ------------------------------------------------------------------------------------------
foreach ($r in @("boreal-api")) {
    if (-not (Exists { aws ecr describe-repositories --repository-names $r --region $Region })) {
        aws ecr create-repository --repository-name $r --region $Region `
            --image-scanning-configuration scanOnPush=true --image-tag-mutability IMMUTABLE | Out-Null
        Write-Host "created ECR repository $r"
    }
    aws ecr put-lifecycle-policy --repository-name $r --region $Region `
        --lifecycle-policy-text (Render "ecr-lifecycle.json") | Out-Null
}
$Registry = "$Account.dkr.ecr.$Region.amazonaws.com"

# 3. App Runner -> ECR access role ----------------------------------------------------------------
if (-not (Exists { aws iam get-role --role-name BorealAppRunnerEcrAccess })) {
    aws iam create-role --role-name BorealAppRunnerEcrAccess `
        --assume-role-policy-document (Render "apprunner-ecr-trust.json") | Out-Null
    aws iam attach-role-policy --role-name BorealAppRunnerEcrAccess `
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
    Write-Host "created role BorealAppRunnerEcrAccess (waiting 10 s for IAM propagation)"; Start-Sleep 10
}
$EcrRole = "arn:aws:iam::${Account}:role/BorealAppRunnerEcrAccess"

# 4. GitHub deploy role (OIDC) --------------------------------------------------------------------
if (-not (Exists { aws iam get-role --role-name BorealGithubDeploy })) {
    aws iam create-role --role-name BorealGithubDeploy --max-session-duration 3600 `
        --assume-role-policy-document (Render "github-oidc-trust.json") | Out-Null
    Write-Host "created role BorealGithubDeploy"
} else {
    aws iam update-assume-role-policy --role-name BorealGithubDeploy --policy-document (Render "github-oidc-trust.json")
}
aws iam put-role-policy --role-name BorealGithubDeploy --policy-name boreal-deploy `
    --policy-document (Render "github-deploy-policy.json")

# 5. First images + services ----------------------------------------------------------------------
$sha = (git rev-parse HEAD).Trim()
$tag = "$sha-bootstrap"   # tags are immutable; CD later pushes the bare commit SHA
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $Registry
docker build -f docker/Dockerfile --build-arg GIT_SHA=$sha --build-arg IMAGE_TAG=$tag -t "$Registry/boreal-api:$tag" .
docker push "$Registry/boreal-api:$tag"

$asName = "boreal-single-instance"
$asArn = aws apprunner list-auto-scaling-configurations --region $Region `
    --auto-scaling-configuration-name $asName --query "AutoScalingConfigurationSummaryList[0].AutoScalingConfigurationArn" --output text
if ($asArn -eq "None" -or -not $asArn) {
    $asArn = aws apprunner create-auto-scaling-configuration --region $Region --auto-scaling-configuration-name $asName `
        --min-size 1 --max-size 1 --max-concurrency 50 --query "AutoScalingConfiguration.AutoScalingConfigurationArn" --output text
}

function Ensure-Service($name, $image, $port, $cpu, $mem, $healthPath, $envVars) {
    $arn = aws apprunner list-services --region $Region --query "ServiceSummaryList[?ServiceName=='$name'].ServiceArn | [0]" --output text
    if ($arn -and $arn -ne "None") { return $arn }
    $src = @{
        ImageRepository = @{ ImageIdentifier = $image; ImageRepositoryType = "ECR";
            ImageConfiguration = @{ Port = "$port"; RuntimeEnvironmentVariables = $envVars } }
        AuthenticationConfiguration = @{ AccessRoleArn = $EcrRole }
        AutoDeploymentsEnabled = $false
    } | ConvertTo-Json -Depth 6 -Compress
    $srcFile = Join-Path $tmp "$name-source.json"; Set-Content $srcFile $src -Encoding ascii
    $hc = "Protocol=HTTP,Path=$healthPath,Interval=10,Timeout=5,HealthyThreshold=1,UnhealthyThreshold=5"
    $arn = aws apprunner create-service --region $Region --service-name $name `
        --source-configuration "file://$($srcFile -replace '\\','/')" `
        --instance-configuration "Cpu=$cpu,Memory=$mem" --health-check-configuration $hc `
        --auto-scaling-configuration-arn $asArn --query "Service.ServiceArn" --output text
    Write-Host "creating App Runner service $name ($arn) - takes ~5 minutes"
    return $arn
}

$apiArn = Ensure-Service "boreal-api" "$Registry/boreal-api:$tag" 8000 $ApiCpu $ApiMemory "/ready" @{ GIT_SHA = $sha; IMAGE_TAG = $tag; BOREAL_API_URL = "http://127.0.0.1:8000" }
do { Start-Sleep 20; $st = aws apprunner describe-service --region $Region --service-arn $apiArn --query "Service.Status" --output text; Write-Host "  boreal-api: $st" } while ($st -eq "OPERATION_IN_PROGRESS")
$apiUrl = "https://" + (aws apprunner describe-service --region $Region --service-arn $apiArn --query "Service.ServiceUrl" --output text)

Write-Host ""
Write-Host "Done. Set these GitHub repository variables (Settings -> Secrets and variables -> Actions -> Variables):"
Write-Host "  AWS_REGION             = $Region"
Write-Host "  AWS_DEPLOY_ROLE_ARN    = arn:aws:iam::${Account}:role/BorealGithubDeploy"
Write-Host "  APPRUNNER_ECR_ROLE_ARN = $EcrRole"
Write-Host "  APPRUNNER_API_ARN      = $apiArn"
Write-Host "  API_URL                = $apiUrl"
Write-Host "  UI_URL                 = <your app URL on Streamlit Community Cloud (runbook 2b)>"
Write-Host "and create the environment 'production' (Settings -> Environments) limited to the master branch."
Write-Host "API docs: $apiUrl/docs   -- next: deploy the interface on Streamlit Community Cloud with API_URL=$apiUrl"
