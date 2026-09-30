"""Resolve the serving model: MLflow registry alias first, local artifact as fallback.

Rollback = move the ``champion`` alias to an earlier registry version; the service picks it up on
restart without a rebuild or redeploy. If the registry is unreachable (e.g. the cloud container has
no MLflow server) the committed ``artifacts/model.joblib`` is used and ``/version`` reports it.
"""

from __future__ import annotations

import os
from typing import Any

import joblib

from boreal_marche.config import Params, Paths
from boreal_marche.retention.artifact import RetentionModel


def load_retention_model(
    params: Params, paths: Paths, alias: str = "champion"
) -> tuple[RetentionModel, str]:
    source = os.environ.get("BOREAL_MODEL_SOURCE", "registry")
    if source == "registry":
        try:
            import mlflow

            from boreal_marche.tracking import tracking_uri

            mlflow.set_tracking_uri(tracking_uri(params, paths))
            uri = f"models:/{params.mlflow.registered_model}@{alias}"
            pyfunc: Any = mlflow.pyfunc.load_model(uri)
            model = pyfunc.unwrap_python_model().model
            if isinstance(model, RetentionModel):
                return model, uri
        except Exception as exc:  # noqa: BLE001 - any registry failure falls back to the file
            print(f"registry unavailable ({type(exc).__name__}); using local artifact")
    return joblib.load(paths.model_file), str(paths.model_file.relative_to(paths.root))
