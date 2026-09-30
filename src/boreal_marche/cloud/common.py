"""Pieces shared by the cloud launchers: one search space and one self-contained source dir."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from boreal_marche.config import Paths

Kind = Literal["int", "float", "logfloat"]

# XGBoost search space for managed HPO (a subset of the local Optuna space in retention/tune.py).
SEARCH_SPACE: dict[str, tuple[Kind, float, float]] = {
    "n-estimators": ("int", 100, 1000),
    "learning-rate": ("logfloat", 0.005, 0.3),
    "max-depth": ("int", 2, 10),
    "min-child-weight": ("logfloat", 1.0, 50.0),
    "subsample": ("float", 0.5, 1.0),
    "colsample-bytree": ("float", 0.4, 1.0),
    "reg-lambda": ("logfloat", 0.001, 10.0),
}
METRIC_NAME = "cv_pr_auc"
METRIC_REGEX = r"cv_pr_auc=([0-9\.]+)"

ENTRY = """import os

os.environ.setdefault("BOREAL_ROOT", os.path.dirname(os.path.abspath(__file__)))

from boreal_marche.retention.sm_train import main  # noqa: E402

if __name__ == "__main__":
    main()
"""
REQUIREMENTS = """pandas>=2.2,<3
pyarrow>=16
pyyaml>=6
pydantic>=2.8,<3
scikit-learn==1.7.2
xgboost>=2.1,<4
lightgbm>=4.3,<5
joblib>=1.4
"""


def log_grid(lo: float, hi: float, n: int = 8) -> list[float]:
    """Log-spaced values (for backends whose Bayesian sampler has no log-uniform prior)."""
    return [
        round(math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / (n - 1)), 6)
        for i in range(n)
    ]


def build_source_dir(paths: Paths | None = None) -> Path:
    """Entry point + a copy of the package + pinned requirements, runnable with ``python -I``."""
    paths = paths or Paths()
    tmp = Path(tempfile.mkdtemp(prefix="boreal_hpo_"))
    shutil.copytree(
        paths.root / "src" / "boreal_marche",
        tmp / "boreal_marche",
        ignore=shutil.ignore_patterns("__pycache__", "cloud", "reporting"),
    )
    shutil.copy(paths.root / "params.yaml", tmp / "params.yaml")
    (tmp / "sm_entry.py").write_text(ENTRY, encoding="utf-8")
    (tmp / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    return tmp


def evidence_template(provider: str) -> dict[str, object]:
    """Fields to record after a real run (reports/metrics/cloud_hpo_evidence_<provider>.json)."""
    return {
        "provider": provider,
        "job_name": "",
        "best_trial": "",
        "best_cv_pr_auc": None,
        "n_trials": None,
        "wall_clock_minutes": None,
        "billable_seconds_total": None,
        "spot_or_low_priority_savings_percent": None,
        "instance_or_vm_size": "",
        "cost_from_billing": None,
        "currency": "USD",
        "date": "YYYY-MM-DD",
    }
