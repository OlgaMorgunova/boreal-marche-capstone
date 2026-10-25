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
        all api clean-mlflow mlflow-ui up down reference assistant-eval simulate loadtest \
        backfill frontend

setup:            ## .venv with Python 3.11 (uv) + latest compatible versions of all extras
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -e ".[train,frontend,loadtest,report,dev]"

setup-pinned:     ## .venv with the exact tested versions (requirements-dev.txt)
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) -e . --no-deps

data etl features dq spotcheck tune train leakage rules-sweep rules rules-eval figures reports \
test cov lint format typecheck api mlflow-ui clean-mlflow all reference index assistant-eval \
loadtest backfill frontend:
	$(RUN) $@

evaluate:         ## one-time test evaluation; `make evaluate ARGS=--force` adds a reported look
	$(RUN) evaluate $(ARGS)

hpo:              ## hyper-parameter search: make hpo PROVIDER=local|aws|azure ARGS=--dry-run
	$(RUN) hpo --provider $(PROVIDER) $(ARGS)

simulate:         ## Locust traffic, normal then shifted: make simulate ARGS="--host https://..."
	$(RUN) simulate $(ARGS)

up:               ## whole stack: API, UI, Postgres, Ollama (docker/compose.yml)
	docker compose -f docker/compose.yml up --build -d

down:
	docker compose -f docker/compose.yml down
