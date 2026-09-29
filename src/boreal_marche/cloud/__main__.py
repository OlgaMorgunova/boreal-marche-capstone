"""Run the hyper-parameter search on the chosen backend.

    python -m boreal_marche.cloud                        # provider from params.yaml (default local)
    python -m boreal_marche.cloud --provider aws   --bucket B --role-arn ARN [--dry-run]
    python -m boreal_marche.cloud --provider azure --subscription S --resource-group G \\
                                   --workspace W [--dry-run]

``local`` runs the Optuna search (retention/tune.py) on this machine: no account, no cost.
Cloud providers print the job definition with ``--dry-run`` and never run from CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from boreal_marche.cloud import aws, azure
from boreal_marche.cloud.common import build_source_dir, evidence_template
from boreal_marche.config import Paths, load_params
from boreal_marche.utils import write_json


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m boreal_marche.cloud")
    ap.add_argument(
        "--provider",
        choices=["local", "aws", "azure"],
        default=os.environ.get("HPO_PROVIDER") or load_params().cloud.provider,
    )
    ap.add_argument("--dry-run", action="store_true", help="print the job definition only")
    ap.add_argument("--bucket", default=os.environ.get("SAGEMAKER_BUCKET"))
    ap.add_argument("--role-arn", default=os.environ.get("SAGEMAKER_ROLE_ARN"))
    ap.add_argument("--subscription", default=os.environ.get("AZURE_SUBSCRIPTION_ID"))
    ap.add_argument("--resource-group", default=os.environ.get("AZURE_RESOURCE_GROUP"))
    ap.add_argument("--workspace", default=os.environ.get("AZUREML_WORKSPACE"))
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse(argv)
    params, paths = load_params(), Paths()
    if a.provider == "local":
        if a.dry_run:
            print(
                json.dumps(
                    {
                        "provider": "local",
                        "service": "Optuna TPE + median pruning",
                        "n_trials_per_model": params.tuning.n_trials,
                        "command": "python -m boreal_marche.retention.tune",
                    },
                    indent=2,
                )
            )
            return 0
        from boreal_marche.retention import tune

        tune.run(params, paths)
        return 0
    if a.provider == "aws":
        missing = [n for n, v in (("--bucket", a.bucket), ("--role-arn", a.role_arn)) if not v]
        if missing and not a.dry_run:
            sys.exit(f"aws needs {', '.join(missing)} (or SAGEMAKER_BUCKET / SAGEMAKER_ROLE_ARN)")
        spec = aws.definition(params, a.bucket or "<bucket>", a.role_arn or "<role-arn>")
    else:
        need = (
            ("--subscription", a.subscription),
            ("--resource-group", a.resource_group),
            ("--workspace", a.workspace),
        )
        missing = [n for n, v in need if not v]
        if missing and not a.dry_run:
            sys.exit(f"azure needs {', '.join(missing)} (or AZURE_* env vars, see .env.example)")
        spec = azure.definition(
            params,
            a.subscription or "<subscription>",
            a.resource_group or "<resource-group>",
            a.workspace or "<workspace>",
        )
    print(json.dumps(spec, indent=2))
    if a.dry_run:
        print("source_dir preview:", build_source_dir(paths))
        return 0
    if not (paths.gold / "retention.parquet").exists():
        sys.exit("data/gold/retention.parquet is missing: run the pipeline first (make features)")
    if a.provider == "aws":  # pragma: no cover - requires AWS
        name = aws.launch(params, paths, a.bucket, a.role_arn)
    else:  # pragma: no cover - requires Azure
        name = azure.launch(params, paths, a.subscription, a.resource_group, a.workspace)
    out = paths.metrics / f"cloud_hpo_evidence_{a.provider}.json"
    write_json(out, evidence_template(a.provider) | {"job_name": name})  # pragma: no cover
    print(f"Started {name}. Fill in {out} from the console when it finishes.")  # pragma: no cover
    return 0


if __name__ == "__main__":
    sys.exit(main())
