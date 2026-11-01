"""Cross-platform task runner (Windows has no `make`). The Makefile delegates here.

    python scripts/run.py all              # full offline pipeline on a fresh clone
    python scripts/run.py train evaluate   # any stages, in order
    python scripts/run.py evaluate --force # re-score the test set (adds a reported look)
    python scripts/run.py hpo --provider azure --dry-run   # extra args go to the last stage
    python scripts/run.py notebook         # re-execute the walkthrough notebook, outputs saved
    python scripts/run.py api              # D4: API on :8000 (second terminal: `frontend`, :8501)
    python scripts/run.py simulate         # D4: Locust traffic, normal then shifted cohort
    python scripts/run.py minimise explain fairness   # D5: ablation, explanations, fairness audit
    python scripts/run.py latency --label after       # D5: latency profile (see the script)
    python scripts/run.py --list

Run it with the project interpreter: `.venv/bin/python` (Linux/macOS) or `.venv\\Scripts\\python`
(Windows). Stages run as subprocesses of that same interpreter, with `src/` on PYTHONPATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M = [sys.executable, "-m"]

STAGES: dict[str, list[list[str]]] = {
    "data": [M + ["boreal_marche.data.ingest"]],
    "etl": [M + ["boreal_marche.data.cleaning"]],
    "features": [M + ["boreal_marche.features"]],
    "dq": [M + ["boreal_marche.data.quality"]],
    "spotcheck": [M + ["boreal_marche.retention.spotcheck"]],
    "tune": [M + ["boreal_marche.retention.tune"]],
    "train": [M + ["boreal_marche.retention.train"]],
    "evaluate": [M + ["boreal_marche.retention.evaluate", "--skip-if-done"]],
    "leakage": [M + ["boreal_marche.retention.leakage"]],
    "rules-sweep": [M + ["boreal_marche.recommend.mine", "sweep"]],
    "rules": [M + ["boreal_marche.recommend.mine"]],
    "rules-eval": [M + ["boreal_marche.recommend.evaluate"]],
    "hpo": [M + ["boreal_marche.cloud"]],
    "reference": [M + ["boreal_marche.monitoring.reference"]],
    "index": [M + ["boreal_marche.assistant.index"]],
    "assistant-eval": [M + ["boreal_marche.assistant.evaluate"]],
    "simulate": [[sys.executable, "scripts/simulate_traffic.py"]],
    "loadtest": [M + ["locust", "-f", "loadtest/locustfile.py", "--host", "http://127.0.0.1:8000"]],
    "backfill": [M + ["boreal_marche.monitoring.backfill", "--replay-validation"]],
    "minimise": [M + ["boreal_marche.retention.minimise"]],
    "explain": [M + ["boreal_marche.retention.explain"]],
    "fairness": [M + ["boreal_marche.retention.fairness"]],
    "latency": [[sys.executable, "scripts/latency_profile.py"]],
    "scalability": [[sys.executable, "scripts/scalability_probe.py"]],
    "frontend": [M + ["streamlit", "run", "frontend/app.py", "--server.port", "8501"]],
    "figures": [M + ["boreal_marche.reporting.figures"]],
    "reports": [M + ["boreal_marche.reporting.figures"], M + ["boreal_marche.reporting.build"]],
    "notebook": [
        M
        + [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=900",
            "notebooks/01_project_walkthrough.ipynb",
        ]
    ],
    "test": [M + ["pytest"]],
    "cov": [
        M + ["pytest", "--cov=boreal_marche", "--cov-report=term-missing", "--cov-fail-under=70"]
    ],
    "lint": [
        M + ["ruff", "check", "."],
        M + ["ruff", "format", "--check", "src", "tests", "scripts"],
    ],
    "format": [M + ["ruff", "format", "src", "tests", "scripts"]],
    "typecheck": [M + ["mypy", "src"]],
    "api": [M + ["uvicorn", "boreal_marche.api:app", "--host", "127.0.0.1", "--port", "8000"]],
    "mlflow-ui": [
        M + ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5000"]
    ],
}
PIPELINE = [
    "data",
    "etl",
    "features",
    "dq",
    "spotcheck",
    "tune",
    "train",
    "evaluate",
    "leakage",
    "rules-sweep",
    "rules",
    "rules-eval",
    "reference",
    "index",
]


def clean_mlflow() -> None:
    (ROOT / "mlflow.db").unlink(missing_ok=True)
    shutil.rmtree(ROOT / "mlruns", ignore_errors=True)
    print("removed mlflow.db and mlruns/")


def run_stage(name: str, extra: list[str], env: dict[str, str]) -> None:
    for cmd in STAGES[name]:
        full = list(cmd)
        if name == "evaluate" and "--force" in extra:
            full.remove("--skip-if-done")
        full += extra if cmd is STAGES[name][-1] else []
        print(
            f"\n=== {name}: {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(full))}",
            flush=True,
        )
        code = subprocess.call(full, cwd=ROOT, env=env)
        if code != 0:
            sys.exit(f"stage '{name}' failed with exit code {code}")


def main(argv: list[str]) -> None:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "--list":
        print("stages:", ", ".join([*STAGES, "all", "clean-mlflow"]))
        print("all =", " -> ".join(PIPELINE))
        return
    names = [a for a in argv if not a.startswith("-")]
    known = set(STAGES) | {"all", "clean-mlflow"}
    stages = [n for n in names if n in known]
    extra = argv[argv.index(stages[-1]) + 1 :] if stages else []
    extra = [a for a in extra if a not in known]
    unknown = [n for n in names if n not in known and n not in extra]
    if unknown or not stages:
        sys.exit(f"unknown stage(s) {unknown or names}; see --list")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    env.setdefault("PYTHONUTF8", "1")  # Windows consoles: UTF-8 for 'Boréal', arrows, etc.
    for n in stages:
        if n == "all":
            for p in PIPELINE:
                run_stage(p, [], env)
        elif n == "clean-mlflow":
            clean_mlflow()
        else:
            run_stage(n, extra if n == stages[-1] else [], env)


if __name__ == "__main__":
    main(sys.argv[1:])
