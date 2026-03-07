# Patient Safety LLM Toolkit

A reproducible toolkit for evaluating risk classification behavior in LLM-assisted clinical text workflows.

## What This Repository Contains

This repository is intentionally scoped to the **LLM toolchain only**:
- Real LLM evaluation workflow (`run_real_llm_study.py`)
- Core Python modules under `src/`
- Automated tests under `tests/`
- Development tooling (`Makefile`, `pyproject.toml`, pre-commit, CI)

Manuscript-editing and publication-asset generation are intentionally excluded from Git tracking.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
```

### 3. Run the main evaluation
```bash
python run_real_llm_study.py --data data/sample_clinical.csv
```

### 4. Run tests
```bash
pytest tests/ -v
```

## Command Guide

Use `make help` to list all commands.

Common commands:
```bash
make check      # lint + typecheck + test
make eval       # run real LLM evaluation
make train      # train LoRA adapter on sample data
make serve      # start local API server
```

## Project Layout

```text
patient_safety_llm/
  README.md                     # Project entry point
  CONTRIBUTING.md               # Contribution workflow
  DEPLOYMENT_GUIDE.md           # Deployment options
  Makefile                      # Standard developer commands
  run_real_llm_study.py         # Main study runner
  src/                          # Core modules
  tests/                        # Test suite
  docs/                         # Study and implementation docs
  notebooks/                    # Walkthrough notebooks
```

For a documentation index, see `docs/README.md`.

## Core Modules (src/)

- `src/llm_evaluation.py`: evaluation orchestration
- `src/llm_interface.py`: provider/model interaction layer
- `src/risk_assessment.py`: risk logic utilities
- `src/evaluate.py`: synthetic/legacy evaluation path
- `src/serve_custom_llm_api.py`: local serving entrypoint
- `src/train_custom_llm_lora.py`: LoRA training workflow

## Quality and CI

Local quality gate:
```bash
make check
```

CI workflow:
- `.github/workflows/tests.yml`

## License

MIT. See `LICENSE`.
