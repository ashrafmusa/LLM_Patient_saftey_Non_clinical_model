# Patient Safety Risks of LLMs in Healthcare

![Status](https://img.shields.io/badge/Status-Submission_Ready-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

**A reproducible pipeline for assessing patient safety risks of Large Language Models in healthcare using synthetic clinical scenarios.**

---

## Quick Start

### Real LLM Study Workflow (Aim 1)
```bash
# Install dependencies
pip install -r requirements.txt

# Original old configuration (kept as default, env-driven)
set OPENAI_API_KEY=your_key
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini

# Run real-LLM evaluation on labeled scenarios
python run_real_llm_study.py --data data/sample_clinical.csv

# One-command provider switch presets
python run_real_llm_study.py --provider grok --data data/sample_clinical.csv
python run_real_llm_study.py --provider ollama --data data/sample_clinical.csv
python run_real_llm_study.py --provider llama_cpp --data data/sample_clinical.csv

# Windows one-command launcher
powershell -ExecutionPolicy Bypass -File .\\run_aim1.ps1 -Provider grok -Data LLM/temp_scenarios.csv

# Optional overrides
python run_real_llm_study.py --provider grok --model grok-2-latest --base-url https://api.x.ai/v1 --data data/sample_clinical.csv

# Execute unit tests
pytest tests/
```

### Custom Project LLM Workflow (Train + Serve)
Use this path when you want a project-specific adapted model instead of API-only prompting.

```bash
# Install base and optional fine-tuning dependencies
pip install -r requirements.txt
pip install peft accelerate

# Train LoRA adapter on your labeled scenarios
python -m src.train_custom_llm_lora \
	--data data/sample_clinical.csv \
	--base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
	--output-dir models/custom_llm_lora

# Serve local inference API
set CUSTOM_LLM_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set CUSTOM_LLM_ADAPTER_PATH=models/custom_llm_lora/adapter
uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010

# Easy operator UI (in a second terminal)
streamlit run src/ui_custom_llm.py
```

Detailed plan and publication steps: `docs/custom_llm_build_deploy_publish_plan.md`.

### Legacy Synthetic Workflow
The previous synthetic-only evaluation path is retained for reproducibility and comparison:

```bash
python src/evaluate.py
```

---

## 📁 Repository Structure

```
README.md                                   ← Project overview (this file)
LICENSE                                     ← MIT license
CONTRIBUTING.md                             ← Contribution guidelines
requirements.txt                            ← Python dependency pinning
src/                                        ← Python modules
tests/                                      ← Test suite
data/                                       ← Data files (gitignored)
models/                                     ← Model artifacts (gitignored)
docs/                                       ← Extended technical documentation
Dockerfile                                  ← Container build recipe
docker-compose.yml                          ← Container orchestration (optional)
```

Only three Markdown files remain at the top level—README, LICENSE, and CONTRIBUTING—to keep the repository focused on the codebase itself. Detailed narrative documentation lives inside `docs/` if your team needs supplemental context.


## 🧪 Test Suite

Run all tests:
```bash
pytest tests/ -v
```

Test coverage:
```bash
pytest tests/ --cov=src
```

Key test modules:
- `tests/test_risk.py`
- `tests/test_llm.py`
- `tests/test_explain.py`
- `tests/test_evaluate.py`

## Clean Code Workflow

This project uses a quality gate with `ruff`, `black`, `mypy`, and `pytest`.

One-time setup:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install
```

Daily workflow:
```bash
# Run all checks
make check

# Or run individual steps
make lint
make typecheck
make test
```

Before opening a PR:
```bash
pre-commit run --all-files
```

Configuration files:
- `pyproject.toml` (tooling configuration)
- `.pre-commit-config.yaml` (git hook automation)
- `Makefile` (standard task runner)
- `.github/workflows/tests.yml` (CI workflow)
- `.github/PULL_REQUEST_TEMPLATE.md` (PR checklist template)


## 🐳 Docker Deployment

Build container:
```bash
docker build -t patient-safety-llm .
```

Run container:
```bash
docker compose up
```


## 📋 Study Details

### Objective
To develop and validate a reproducible pipeline for assessing patient safety risks of AI-assisted clinical systems, with direct evaluation against real LLM outputs as the primary study path.

### Methods
- Synthetic dataset of 200 adjudicated scenarios stratified into low, medium, and high risk, with augmentation expanding the corpus to 800 samples.
- Text features extracted using TF-IDF (unigrams + bigrams) and evaluated with calibrated logistic regression.
- Five-fold stratified cross-validation ensures robust internal validation, while an external hold-out set highlights generalization gaps.
- Explainability provided through coefficient analysis to surface high-signal clinical terms driving each risk prediction.

### Key Findings
- Perfect performance on augmented synthetic data (accuracy 100%) does not translate to the original evaluation set (accuracy 33%), underscoring the need for external validation.
- The pipeline surfaces systematic bias toward high-risk classifications when distribution shift occurs between training and evaluation cohorts.
- Explainability artifacts enable clinicians to review the terms driving model decisions before deployment.

### Implications
- Healthcare organizations should require external validation before approving LLM-powered clinical tools.
- Risk-stratified evaluation frameworks make distribution shifts visible and measurable.
- Transparent, reproducible tooling accelerates safety assessments and regulatory alignment.


## 📄 License

MIT License - See [LICENSE](LICENSE) for details


## 👤 Author

**Ashraf Abubaker Musa**  
College of Medical Laboratories Sciences  
Alneelain University  
Khartoum, Sudan

Email: Ashraf0968491090@gmail.com
