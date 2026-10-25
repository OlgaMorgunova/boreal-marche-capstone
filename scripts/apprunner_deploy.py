"""Point an existing App Runner service at a new image and wait for the deployment (CD helper).

    python scripts/apprunner_deploy.py --service-arn arn:aws:apprunner:... \
        --image <ecr>/boreal-api:<sha> --port 8000 \
        --access-role-arn arn:aws:iam::...:role/BorealAppRunnerEcrAccess --env KEY=VALUE ...

Uses the AWS CLI (pre-installed on GitHub runners) with the OIDC session credentials of the job.
Auto-deployment is disabled on the services: an image only goes live through this pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time


def aws(*args: str) -> dict:  # type: ignore[type-arg]
    out = subprocess.run(["aws", *args, "--output", "json"], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"aws {' '.join(args[:2])} failed: {out.stderr.strip()}")
    return json.loads(out.stdout or "{}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service-arn", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--access-role-arn", required=True)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE runtime variable")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    env = dict(kv.split("=", 1) for kv in a.env)
    source = {
        "ImageRepository": {
            "ImageIdentifier": a.image,
            "ImageRepositoryType": "ECR",
            "ImageConfiguration": {"Port": a.port, "RuntimeEnvironmentVariables": env},
        },
        "AuthenticationConfiguration": {"AccessRoleArn": a.access_role_arn},
        "AutoDeploymentsEnabled": False,
    }
    op = aws(
        "apprunner", "update-service", "--service-arn", a.service_arn,
        "--source-configuration", json.dumps(source),
    )["OperationId"]  # fmt: skip
    print(f"update-service started: operation {op} -> {a.image}", flush=True)
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        ops = aws("apprunner", "list-operations", "--service-arn", a.service_arn)
        status = next((o["Status"] for o in ops["OperationSummaryList"] if o["Id"] == op), "?")
        print(f"  {time.strftime('%H:%M:%S')} {status}", flush=True)
        if status == "SUCCEEDED":
            svc = aws("apprunner", "describe-service", "--service-arn", a.service_arn)["Service"]
            print(f"deployed: https://{svc['ServiceUrl']} ({svc['Status']})")
            return
        if status in ("FAILED", "ROLLBACK_SUCCEEDED", "ROLLBACK_FAILED"):
            sys.exit(f"deployment {status}: see the App Runner service event log")
        time.sleep(15)
    sys.exit("timed out waiting for the App Runner deployment")


if __name__ == "__main__":
    main()
