"""Amazon SageMaker Bayesian HPO job (managed spot) — the backend named in the course spec."""

from __future__ import annotations

from typing import Any

from boreal_marche.cloud.common import METRIC_NAME, METRIC_REGEX, SEARCH_SPACE, build_source_dir
from boreal_marche.config import Params, Paths


def definition(params: Params, bucket: str, role_arn: str) -> dict[str, Any]:
    a, c = params.cloud.aws, params.cloud
    return {
        "provider": "aws",
        "service": "SageMaker HyperparameterTuner",
        "region": a.region,
        "instance_type": a.instance_type,
        "use_spot_instances": a.use_spot,
        "max_run": a.max_run_seconds,
        "max_wait": a.max_wait_seconds,
        "max_jobs": c.n_trials,
        "max_parallel_jobs": c.max_parallel,
        "strategy": "Bayesian",
        "objective": {"Name": METRIC_NAME, "Regex": METRIC_REGEX, "type": "Maximize"},
        "ranges": {k: {"kind": v[0], "min": v[1], "max": v[2]} for k, v in SEARCH_SPACE.items()},
        "train_s3": f"s3://{bucket}/gold/",
        "output_s3": f"s3://{bucket}/hpo-output/",
        "role": role_arn,
        "framework_version": a.framework_version,
    }


def launch(params: Params, paths: Paths, bucket: str, role_arn: str) -> str:  # pragma: no cover
    import boto3
    import sagemaker
    from sagemaker.sklearn.estimator import SKLearn
    from sagemaker.tuner import ContinuousParameter, HyperparameterTuner, IntegerParameter

    a, c = params.cloud.aws, params.cloud
    session = sagemaker.Session(
        boto_session=boto3.Session(region_name=a.region), default_bucket=bucket
    )
    train_s3 = session.upload_data(str(paths.gold / "retention.parquet"), key_prefix="gold")
    est = SKLearn(
        entry_point="sm_entry.py",
        source_dir=str(build_source_dir(paths)),
        framework_version=a.framework_version,
        py_version="py3",
        role=role_arn,
        instance_type=a.instance_type,
        instance_count=1,
        use_spot_instances=a.use_spot,
        max_run=a.max_run_seconds,
        max_wait=a.max_wait_seconds,
        sagemaker_session=session,
        output_path=f"s3://{bucket}/hpo-output/",
    )
    ranges: dict[str, Any] = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            ranges[k] = IntegerParameter(int(lo), int(hi))
        else:
            ranges[k] = ContinuousParameter(
                lo, hi, scaling_type="Logarithmic" if kind == "logfloat" else "Linear"
            )
    tuner = HyperparameterTuner(
        est,
        objective_metric_name=METRIC_NAME,
        metric_definitions=[{"Name": METRIC_NAME, "Regex": METRIC_REGEX}],
        hyperparameter_ranges=ranges,
        strategy="Bayesian",
        objective_type="Maximize",
        max_jobs=c.n_trials,
        max_parallel_jobs=c.max_parallel,
    )
    tuner.fit({"train": train_s3.rsplit("/", 1)[0]}, wait=False)
    return str(tuner.latest_tuning_job.name)
