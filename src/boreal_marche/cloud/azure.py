"""Azure Machine Learning sweep job (Bayesian sampling, low-priority compute).

Azure's Bayesian sampler supports only ``choice``, ``uniform`` and ``quniform``: integer ranges use
``QUniform(q=1)`` and log-scaled ranges become a log-spaced ``Choice`` grid. Metrics reach the sweep
through MLflow (``--log-mlflow``), which Azure ML wires up automatically inside the job.
"""

from __future__ import annotations

import math
from typing import Any

from boreal_marche.cloud.common import METRIC_NAME, SEARCH_SPACE, build_source_dir, log_grid
from boreal_marche.config import Params, Paths

CONDA = """name: boreal-train
channels: [conda-forge]
dependencies:
  - python=3.11
  - pip
  - pip:
{pins}
      - mlflow>=2.16,<3
      - azureml-mlflow
"""
BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"


def search_space() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            out[k] = {"type": "quniform", "min_value": lo, "max_value": hi, "q": 1}
        elif kind == "float":
            out[k] = {"type": "uniform", "min_value": lo, "max_value": hi}
        else:
            out[k] = {"type": "choice", "values": log_grid(lo, hi)}
    return out


def command_line() -> str:
    args = " ".join(f"--{k} ${{{{inputs.{k.replace('-', '_')}}}}}" for k in SEARCH_SPACE)
    return (
        f"python sm_entry.py --train ${{{{inputs.train}}}} --model-dir ${{{{outputs.model}}}} "
        f"--log-mlflow {args}"
    )


def definition(
    params: Params, subscription: str, resource_group: str, workspace: str
) -> dict[str, Any]:
    z, c = params.cloud.azure, params.cloud
    return {
        "provider": "azure",
        "service": "Azure ML sweep job",
        "location": z.location,
        "workspace": f"{subscription}/{resource_group}/{workspace}",
        "compute": {
            "name": z.compute_name,
            "size": z.vm_size,
            "tier": z.tier,
            "min_instances": 0,
            "max_instances": c.max_parallel,
            "idle_seconds_before_scale_down": z.idle_seconds_before_scale_down,
        },
        "sampling_algorithm": "bayesian",
        "primary_metric": METRIC_NAME,
        "goal": "Maximize",
        "limits": {
            "max_total_trials": c.n_trials,
            "max_concurrent_trials": c.max_parallel,
            "timeout": z.timeout_seconds,
        },
        "search_space": search_space(),
        "command": command_line(),
        "environment": {"image": BASE_IMAGE, "python": "3.11"},
    }


def launch(
    params: Params, paths: Paths, subscription: str, resource_group: str, workspace: str
) -> str:  # pragma: no cover - requires Azure credentials
    from azure.ai.ml import Input, MLClient, Output, command
    from azure.ai.ml.entities import AmlCompute, Environment
    from azure.ai.ml.sweep import Choice, QUniform, Uniform
    from azure.identity import DefaultAzureCredential

    z, c = params.cloud.azure, params.cloud
    ml = MLClient(DefaultAzureCredential(), subscription, resource_group, workspace)
    ml.compute.begin_create_or_update(
        AmlCompute(
            name=z.compute_name,
            size=z.vm_size,
            tier=z.tier,
            min_instances=0,
            max_instances=c.max_parallel,
            idle_time_before_scale_down=z.idle_seconds_before_scale_down,
        )
    ).result()
    src = build_source_dir(paths)
    pins = "\n".join(
        f"      - {line}" for line in (src / "requirements.txt").read_text().splitlines() if line
    )
    (src / "conda.yaml").write_text(CONDA.format(pins=pins), encoding="utf-8")
    env = Environment(name="boreal-train", image=BASE_IMAGE, conda_file=str(src / "conda.yaml"))
    defaults = {
        k.replace("-", "_"): (lo if kind != "logfloat" else math.sqrt(lo * hi))
        for k, (kind, lo, hi) in SEARCH_SPACE.items()
    }
    job = command(
        code=str(src),
        command=command_line(),
        environment=env,
        compute=z.compute_name,
        experiment_name=z.experiment,
        inputs={"train": Input(type="uri_folder", path=str(paths.gold)), **defaults},
        outputs={"model": Output(type="uri_folder")},
    )
    space: dict[str, Any] = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        key = k.replace("-", "_")
        if kind == "int":
            space[key] = QUniform(min_value=lo, max_value=hi, q=1)
        elif kind == "float":
            space[key] = Uniform(min_value=lo, max_value=hi)
        else:
            space[key] = Choice(values=log_grid(lo, hi))
    sweep = job(**space).sweep(
        compute=z.compute_name,
        sampling_algorithm="bayesian",
        primary_metric=METRIC_NAME,
        goal="Maximize",
    )
    sweep.set_limits(
        max_total_trials=c.n_trials, max_concurrent_trials=c.max_parallel, timeout=z.timeout_seconds
    )
    created = ml.jobs.create_or_update(sweep)
    return str(created.name)
