# Patient Safety LLM Toolkit

A clinical text risk-classification and evaluation toolkit focused on patient safety workflows, synthetic scenario benchmarking, and safety-oriented LLM assessment.

This repository contains the application code, evaluation pipeline, APIs, tests, and developer tooling for the LLM system. It does not track manuscript-generation assets in GitHub scope.

## Audit Summary

This README reflects the current codebase as audited on the local project state.

What is present today:
- Real-LLM evaluation runner for labeled scenarios
- Synthetic evaluation pipeline and baseline ML workflow
- Two FastAPI app surfaces
- Deterministic patient-safety guardrails around LLM outputs
- Lightweight local retrieval grounding over internal documentation
- JSONL audit logging for `/assess` requests
- Training path for a custom LoRA adapter
- Unit tests and developer tooling

What is not present today:
- Production-grade RAG or vector retrieval layer
- Clinical guideline citation engine
- Authentication and authorization for APIs
- Full governance, audit log, or model registry layer
- Real-world validated clinical incident corpus in the repository

## Repository Scope

This repository is intentionally scoped to the LLM toolchain and application logic:
- Evaluation workflows
- Risk assessment logic
- LLM integration and prompting
- Safety guardrails
- APIs and training scripts
- Tests and developer tooling

Publication and manuscript-editing tooling may exist locally in ignored paths, but this README documents only the application repository scope.

## Core Capabilities

- Evaluate clinical scenarios with multiple LLM prompting strategies
- Compare traditional baseline behavior with LLM-based classification
- Add deterministic safety overrides for high-harm scenarios
- Run a minimal assessment API for integration experiments
- Run a custom fine-tuned generation API using a base model plus optional LoRA adapter
- Produce machine-readable evaluation outputs in CSV and JSON

## Current Safety Model

The current architecture is a layered safety pipeline rather than a raw chatbot:

1. Scenario input
2. Prompted LLM classification
3. Lightweight retrieval grounding for safety-focused prompts
4. Structured parsing of risk result
5. Deterministic input and output guardrails
6. Evaluation metrics for accuracy, parse rate, escalation, uncertainty, and guardrail intervention

The guardrail layer currently enforces conservative escalation for scenarios with strong harm signals such as:
- medication dose mismatch
- anaphylaxis or allergy risk
- unresponsiveness
- self-harm or overdose
- severe deterioration terms such as sepsis or cardiac arrest

The retrieval layer currently grounds safety-focused prompts against local documents under `docs/`. It is a scaffold for future migration to a vector database such as Chroma, FAISS, Qdrant, or Weaviate.

## Main Workflows

### 1. Real LLM evaluation

Run the real model against labeled clinical scenarios:

```bash
python run_real_llm_study.py --data data/sample_clinical.csv
```

Available strategies:
- `zero_shot`
- `few_shot`
- `chain_of_thought`
- `safety_focused`

Example:

```bash
python run_real_llm_study.py --provider openai --strategies safety_focused few_shot --data data/sample_clinical.csv
```

Output:
- `reports/real_llm_eval_<timestamp>/summary.json`
- `reports/real_llm_eval_<timestamp>/results.csv`

### 2. Synthetic baseline evaluation

Run the legacy synthetic pipeline, which can bootstrap a classical model if none exists:

```bash
python -m src.evaluate --n 200 --output reports/eval
```

This path generates scenarios, preprocesses them, evaluates a baseline risk model, and writes metrics and per-case outputs.

### 3. Minimal assessment API

Serve the lightweight assessment API:

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Endpoint:
- `POST /assess`

Behavior:
- retrieves local safety context for grounding
- runs safety-focused LLM classification
- performs baseline model assessment for comparison
- writes a structured JSONL audit record for each request

### 4. Custom model inference API

Serve a custom fine-tuned model endpoint:

```bash
uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010
```

Endpoints:
- `GET /health`
- `POST /predict`

This path loads a base model and optionally a PEFT/LoRA adapter when adapter artifacts are present.

### 5. Custom LoRA training

Train a project-specific adapter:

```bash
python -m src.train_custom_llm_lora --data data/sample_clinical.csv --output-dir models/custom_llm_lora
```

## Setup

### Python

Recommended Python version:
- 3.11

### Install runtime dependencies

```bash
pip install -r requirements.txt
```

### Install developer dependencies

```bash
pip install -r requirements-dev.txt
```

### Environment variables

OpenAI-compatible setup:

```bash
set OPENAI_API_KEY=your_api_key_here
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
```

Local llama.cpp server setup:

```bash
set LLAMA_SERVER_URL=http://127.0.0.1:8080
```

Retrieval and audit setup:

```bash
set RETRIEVAL_ENABLED=true
set RETRIEVAL_DOCS_DIR=docs
set RETRIEVAL_MAX_CHUNKS=3
set AUDIT_LOG_DIR=reports/audit_logs
```

Custom model serving setup:

```bash
set CUSTOM_LLM_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set CUSTOM_LLM_ADAPTER_PATH=models/custom_llm_lora/adapter
```

## Developer Commands

If `make` is available:

```bash
make help
make install
make install-dev
make format
make lint
make typecheck
make test
make check
make eval
make train
make serve
```

On Windows systems without `make`, run the underlying commands directly:

```bash
python -m pytest -q -o addopts='' tests
ruff check src tests
black --check src tests
mypy src
python run_real_llm_study.py --data data/sample_clinical.csv
uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010
```

## API Examples

### `POST /assess`

Request:

```json
{
  "text": "Medication error: ordered 10mg warfarin instead of 1mg. Patient found unresponsive."
}
```

Response shape:

```json
{
  "risk_level": "high",
  "details": {
    "input_text": "Medication error: ordered 10mg warfarin instead of 1mg. Patient found unresponsive.",
    "retrieval": {
      "context": "Source: aim1_real_llm_plan.md\n...",
      "sources": [
        {
          "path": ".../docs/aim1_real_llm_plan.md",
          "score": 0.67,
          "excerpt": "..."
        }
      ],
      "retrieval_enabled": true
    },
    "llm_assessment": {
      "risk_level": "high",
      "event_type": "medication",
      "care_setting": "unknown",
      "action_urgency": "emergent",
      "needs_escalation": true,
      "uncertainty_flag": false,
      "guardrail_triggered": true,
      "guardrail_reasons": ["high_harm_signal", "dose_mismatch"]
    },
    "baseline_assessment": {
      "risk_level": "high",
      "score": 0.91,
      "model_based": true
    },
    "audit": {
      "audit_id": "...",
      "audit_log_path": ".../reports/audit_logs/assess_log.jsonl"
    }
  }
}
```

Audit output:
- `reports/audit_logs/assess_log.jsonl`

### `POST /predict`

Request:

```json
{
  "text": "Medication error: ordered 10mg warfarin instead of 1mg. Patient found unresponsive.",
  "max_new_tokens": 180
}
```

Response shape:

```json
{
  "risk_level": "high",
  "reasoning": "Tenfold dosing error with serious harm potential.",
  "raw_response": "{...}"
}
```

## Prompting Strategies

The LLM evaluation module supports multiple prompt styles:

- `zero_shot`: minimal instruction, structured output requested
- `few_shot`: includes worked examples
- `chain_of_thought`: encourages explicit reasoning before final output
- `safety_focused`: requests structured safety fields such as event type, care setting, urgency, escalation, and uncertainty

The `safety_focused` strategy is the current best option for healthcare-facing evaluations in this repository.

## Safety Guardrails

The guardrail layer is implemented in `src/safety_guardrails.py` and currently provides:

- input prompt-injection detection
- output escalation for severe harm signals
- medication dose mismatch detection using numeric dose extraction
- allergy risk escalation
- guardrail intervention metadata in evaluation results

Guardrail-related summary metrics include:
- `guardrail_trigger_rate`
- `escalation_rate`
- `uncertainty_rate`
- `conservative_bias`

## Project Layout

```text
patient_safety_llm/
  README.md
  CONTRIBUTING.md
  DEPLOYMENT_GUIDE.md
  Makefile
  pyproject.toml
  requirements.txt
  requirements-dev.txt
  run_real_llm_study.py
  src/
    app.py
    config.py
    evaluate.py
    llm_evaluation.py
    llm_interface.py
    risk_assessment.py
    safety_guardrails.py
    serve_custom_llm_api.py
    train_custom_llm_lora.py
    ...
  tests/
  docs/
  notebooks/
  data/
  models/
  reports/
```

## Important Modules

- `src/app.py`
  Minimal FastAPI assessment endpoint for quick integration experiments.

- `src/serve_custom_llm_api.py`
  FastAPI service for a custom model or LoRA adapter.

- `src/llm_interface.py`
  Provider abstraction over OpenAI-compatible APIs, local llama.cpp server, or local binary fallback.

- `src/llm_evaluation.py`
  Prompting strategies, structured parsing, LLM evaluation loop, and summary metrics.

- `src/retrieval.py`
  Local retrieval scaffold for grounded context injection.

- `src/safety_guardrails.py`
  Deterministic input and output safety rules.

- `src/audit_logging.py`
  JSONL audit trail writer for assessment calls.

- `src/risk_assessment.py`
  Classical risk model and heuristic fallback path.

- `src/evaluate.py`
  Synthetic baseline evaluation harness.

## Tests and Quality

Run focused tests:

```bash
python -m pytest -q -o addopts='' tests/test_llm.py
```

Run the repository test suite with configured defaults:

```bash
pytest tests
```

Quality tools configured in `pyproject.toml`:
- Ruff
- Black
- Mypy
- Pytest

CI workflow:
- `.github/workflows/tests.yml`

## Current Limitations

This project is research and prototyping code. It is not a certified medical device.

Known limitations:
- primary included dataset is small and synthetic
- no clinician-adjudicated gold-standard incident set is included in the repository
- retrieval is lexical and local-document based, not vector-search backed
- retrieval does not yet provide standards-grade inline citations or ranking by clinical authority
- audit logging exists, but there is no governance dashboard, review queue, or model registry yet
- APIs do not currently include authentication, authorization, or rate limiting
- `/assess` combines retrieval, LLM assessment, guardrails, and a baseline model in one response, which is useful for experimentation but should be separated into clearer service boundaries before production use

## Recommended Next Steps

The most valuable next engineering upgrades are:

1. Add retrieval and source citation for patient-safety standards and policies.
2. Add clinician-reviewed benchmark data with event taxonomies and escalation labels.
3. Add unsafe false negative metrics and under-triage monitoring.
4. Add API authentication, audit logging, and decision trace storage.
5. Separate research endpoints from production-style safety services.

## Documentation

Documentation index:
- `docs/README.md`

Useful docs:
- `docs/aim1_real_llm_plan.md`
- `docs/plan.md`
- `docs/custom_llm_build_deploy_publish_plan.md`
- `DEPLOYMENT_GUIDE.md`

## License

MIT. See `LICENSE`.
