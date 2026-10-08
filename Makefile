.PHONY: setup lint typecheck test security dataset ingest build-index evaluate benchmark serve clean

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn
PYTEST = $(VENV)/bin/pytest
RUFF = $(VENV)/bin/ruff
MYPY = $(VENV)/bin/mypy
BANDIT = $(VENV)/bin/bandit

setup:
	python3.13 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch
	$(PIP) install -r requirements-dev.txt

lint:
	$(RUFF) check .

typecheck:
	$(MYPY) app

test:
	$(PYTEST) tests/unit tests/integration tests/api tests/security -v --cov=app --cov-report=term-missing

security:
	$(BANDIT) -r app -ll

dataset:
	$(PYTHON) scripts/generate_dataset.py --documents 500000 --output data/raw

ingest:
	$(PYTHON) scripts/ingest.py --input data/raw --output data/processed

build-index:
	$(PYTHON) scripts/build_index.py --input data/processed --output data/indexes

evaluate:
	$(PYTHON) scripts/evaluate.py --index data/indexes/current --output benchmarks

benchmark:
	$(PYTHON) scripts/benchmark.py --concurrency 10,50,100,250,500 --output benchmarks

serve:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --workers 1

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
