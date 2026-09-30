"""MLflow helpers: one tracking URI, common lineage tags on every run."""

from __future__ import annotations

import os
from typing import Any

import mlflow

from boreal_marche.config import Params, Paths
from boreal_marche.utils import git_commit, read_json, sha256_file


def tracking_uri(params: Params, paths: Paths) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI") or params.mlflow.tracking_uri
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{(paths.root / uri.removeprefix('sqlite:///')).as_posix()}"
    return uri


def setup(params: Params, paths: Paths, experiment: str) -> str:
    mlflow.set_tracking_uri(tracking_uri(params, paths))
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = mlflow.create_experiment(
            experiment, artifact_location=(paths.root / "mlruns").as_uri()
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)
    return exp_id


def lineage_tags(paths: Paths) -> dict[str, Any]:
    tags: dict[str, Any] = {"git_commit": git_commit()}
    if paths.manifest.exists():
        tags["raw_sha256"] = read_json(paths.manifest).get("sha256", "unknown")
    gold = paths.gold / "retention.parquet"
    if gold.exists():
        tags["gold_sha256"] = sha256_file(gold)
    return tags
