"""Cloud HPO backends: definitions, search-space mapping and the shared entry point (no network)."""

from __future__ import annotations

import json

import pytest

from boreal_marche.cloud import __main__ as cli
from boreal_marche.cloud import aws, azure
from boreal_marche.cloud.common import SEARCH_SPACE, build_source_dir, log_grid
from boreal_marche.config import load_params
from boreal_marche.retention.sm_train import int_arg, parse_args


@pytest.mark.parametrize("provider", ["local", "aws", "azure"])
def test_dry_run_prints_definition(provider: str, capsys) -> None:  # type: ignore[no-untyped-def]
    assert cli.main(["--provider", provider, "--dry-run"]) == 0
    out = capsys.readouterr().out
    spec = json.loads(out[: out.rindex("}") + 1])
    assert spec["provider"] == provider


def test_aws_definition_caps_cost() -> None:
    spec = aws.definition(load_params(), "bucket", "arn:role")
    assert spec["use_spot_instances"] is True and spec["max_wait"] >= spec["max_run"]
    assert spec["strategy"] == "Bayesian" and set(spec["ranges"]) == set(SEARCH_SPACE)


def test_azure_search_space_uses_supported_distributions() -> None:
    space = azure.search_space()
    assert {v["type"] for v in space.values()} <= {"choice", "uniform", "quniform"}
    assert space["n-estimators"]["q"] == 1
    cmd = azure.command_line()
    assert "--log-mlflow" in cmd and "${{inputs.learning_rate}}" in cmd
    spec = azure.definition(load_params(), "s", "g", "w")
    assert spec["compute"]["min_instances"] == 0 and spec["compute"]["tier"] == "low_priority"


def test_missing_credentials_exit() -> None:
    with pytest.raises(SystemExit):
        cli.main(["--provider", "aws", "--bucket", "", "--role-arn", ""])
    with pytest.raises(SystemExit):
        cli.main(
            ["--provider", "azure", "--subscription", "", "--resource-group", "", "--workspace", ""]
        )


def test_log_grid_and_int_args() -> None:
    g = log_grid(0.01, 1.0, 3)
    assert g == [0.01, 0.1, 1.0]
    assert int_arg("300.0") == 300
    a = parse_args(["--n-estimators", "250.0", "--max-depth", "5"])
    assert a.n_estimators == 250 and a.max_depth == 5 and a.log_mlflow is False


def test_source_dir_is_self_contained() -> None:
    src = build_source_dir()
    assert (src / "sm_entry.py").exists() and (src / "requirements.txt").exists()
    assert (src / "boreal_marche" / "retention" / "sm_train.py").exists()
    assert not (src / "boreal_marche" / "cloud").exists()
