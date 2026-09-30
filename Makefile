# Boréal Marché — one entry point per pipeline stage (assignment §7.2).
# Every target delegates to scripts/run.py, which also works on Windows without make:
#   python scripts/run.py all        (Windows: .venv\Scripts\python scripts\run.py all)
# Usage: make setup && make all && make test && make reports
PY ?= .venv/bin/python
RUN = $(PY) scripts/run.py
PROVIDER ?= local
ARGS ?=

.PHONY: setup setup-pinned data etl features dq spotcheck tune train evaluate leakage \
        rules-sweep rules rules-eval hpo index test cov lint format typecheck figures reports \
        all api clean-mlflow mlflow-ui up

setup:            ## .venv with Python 3.11 (uv) + latest compatible versions of all extras
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -e ".[train,report,dev]"

setup-pinned:     ## .venv with the exact tested versions (requirements-dev.txt)
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) -e . --no-deps

data etl features dq spotcheck tune train leakage rules-sweep rules rules-eval figures reports \
test cov lint format typecheck api mlflow-ui clean-mlflow all:
	$(RUN) $@

evaluate:         ## one-time test evaluation; `make evaluate ARGS=--force` adds a reported look
	$(RUN) evaluate $(ARGS)

hpo:              ## hyper-parameter search: make hpo PROVIDER=local|aws|azure ARGS=--dry-run
	$(RUN) hpo --provider $(PROVIDER) $(ARGS)

index:            ## Head C retrieval index (Week 9 — not implemented yet)
	@echo "Head C index is scheduled for Week 9 (see PLAN.md)"; exit 1

up:
	docker compose -f docker/compose.yml up --build
