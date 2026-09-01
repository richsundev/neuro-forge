.PHONY: setup test lint typecheck reproduce docker-up docker-down docker-build \
	failure-worker-crash failure-evaluator-timeout failure-model-timeout \
	failure-invalid-candidate failure-budget-exceeded failure-canary-regression \
	failure-all migrate api worker dashboard

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]" --python .venv/bin/python
	uv pip install -e ./apps/api --python .venv/bin/python
	cd apps/dashboard && npm install

test:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests apps
	cd apps/dashboard && npm run lint

typecheck:
	$(MYPY) src apps/api/neuroforge_api
	cd apps/dashboard && npx tsc --noEmit

reproduce:
	$(PYTHON) scripts/reproduce.py

migrate:
	.venv/bin/alembic upgrade head

api:
	.venv/bin/uvicorn neuroforge_api.main:app --reload

worker:
	$(PYTHON) scripts/worker.py

dashboard:
	cd apps/dashboard && npm run dev

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down -v

# --- failure injection (section 51) -----------------------------------

failure-worker-crash:
	$(PYTHON) scripts/failures/worker_crash.py

failure-evaluator-timeout:
	$(PYTEST) -q tests/test_experiment_engine.py -k evaluation_timeout -v

failure-model-timeout:
	$(PYTHON) scripts/failures/model_timeout.py

failure-invalid-candidate:
	$(PYTEST) -q tests/test_experiment_engine.py -k outside_mutation_policy -v

failure-budget-exceeded:
	$(PYTEST) -q tests/test_experiment_engine.py -k budget -v

failure-canary-regression:
	$(PYTEST) -q tests/test_promotion.py -k canary -v

failure-all: failure-worker-crash failure-evaluator-timeout failure-model-timeout \
	failure-invalid-candidate failure-budget-exceeded failure-canary-regression
