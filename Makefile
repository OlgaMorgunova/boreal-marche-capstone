setup:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src

etl:
	python -m boreal_marche.etl

rules:
	python -m boreal_marche.rules

train:
	python -m boreal_marche.train

index:
	python -m boreal_marche.index

spotcheck:
	python -m boreal_marche.spotcheck

up:
	docker compose -f docker/compose.yml up --build
