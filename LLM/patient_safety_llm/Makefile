.PHONY: help install install-dev format lint typecheck test check precommit-install precommit-run train eval serve

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

help:
	@echo "Available targets:"
	@echo "  install            Install runtime dependencies"
	@echo "  install-dev        Install runtime + development dependencies"
	@echo "  format             Format code with ruff and black"
	@echo "  lint               Lint code with ruff and black --check"
	@echo "  typecheck          Run mypy on src/"
	@echo "  test               Run pytest"
	@echo "  check              Run lint + typecheck + test"
	@echo "  precommit-install  Install pre-commit hooks"
	@echo "  precommit-run      Run pre-commit on all files"
	@echo "  train              Train LoRA adapter on sample data"
	@echo "  eval               Run real-LLM evaluation"
	@echo "  serve              Start local inference API"

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

