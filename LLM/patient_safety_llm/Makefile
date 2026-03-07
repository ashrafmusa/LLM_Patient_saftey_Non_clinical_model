.PHONY: install install-dev format lint typecheck test check precommit-install precommit-run train eval serve publish

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

format:
	ruff format src tests
	black src tests

lint:
	ruff check src tests
	black --check src tests

typecheck:
	mypy src

test:
	pytest tests

check: lint typecheck test

precommit-install:
	pre-commit install

precommit-run:
	pre-commit run --all-files

train:
	$(PYTHON) -m src.train_custom_llm_lora --data data/sample_clinical.csv --output-dir models/custom_llm_lora

eval:
	$(PYTHON) run_real_llm_study.py --data data/sample_clinical.csv

serve:
	uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010

publish:
	$(PYTHON) tools/publication/generate_revised_figures.py
	$(PYTHON) tools/publication/generate_revised_manuscript.py
	$(PYTHON) tools/publication/generate_response_to_reviewers.py
