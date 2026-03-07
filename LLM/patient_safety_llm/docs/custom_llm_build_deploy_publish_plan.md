# Custom LLM Build, Deploy, and Publication Plan

## Goal
Build a dedicated project-specific LLM that classifies patient safety risk on clinical scenarios, deploy it locally or in cloud infrastructure, and produce publication-grade evidence.

## Why this direction
- The current repository already has strong evaluation and publication infrastructure.
- The linked `xbeat/Machine-Learning` repository is primarily tutorial content and examples.
- This project needs execution artifacts, reproducibility, and reviewer-ready evidence.

## Scope and outputs
1. A fine-tuned model adapter focused on your domain labels (`low`, `medium`, `high`).
2. A local inference API for controlled deployment.
3. Reproducible evaluation reports and figures.
4. Manuscript and response-to-reviewers package using existing publication tools.

## What was added in this repo
- `src/train_custom_llm_lora.py`: trains a LoRA adapter from labeled CSV data.
- `src/serve_custom_llm_api.py`: serves the tuned model with FastAPI.

## Data contract
Input CSV must include:
- `text`: scenario narrative.
- `label`: one of `low`, `medium`, `high`.

Recommended minimum:
- Pilot: 300+ labeled rows.
- Publication target: 1000+ labeled rows with expert adjudication.

## Build workflow
### 1. Prepare environment
```bash
pip install -r requirements.txt
pip install peft accelerate
```

Optional for memory-constrained GPUs:
```bash
pip install bitsandbytes
```

### 2. Train custom LoRA adapter
```bash
python -m src.train_custom_llm_lora \
  --data data/sample_clinical.csv \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir models/custom_llm_lora
```

Optional 4-bit quantized training:
```bash
python -m src.train_custom_llm_lora \
  --data data/sample_clinical.csv \
  --output-dir models/custom_llm_lora \
  --use-4bit
```

### 3. Evaluate against your existing benchmark harness
```bash
python run_real_llm_study.py --provider llama_cpp --data data/sample_clinical.csv
```

If you want direct adapter evaluation inside this repo, expose the model through the API below and point your evaluation stage to that endpoint.

## Deployment workflow
### Local API deployment
```bash
set CUSTOM_LLM_BASE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
set CUSTOM_LLM_ADAPTER_PATH=models/custom_llm_lora/adapter
uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010
```

### Smoke test
```bash
curl -X POST http://127.0.0.1:8010/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Medication dose was entered ten times higher than ordered.\"}"
```

## Publication workflow
1. Lock the dataset version used for training and evaluation.
2. Save `training_metadata.json` from model output directory.
3. Save evaluation outputs under `reports/` with timestamped directories.
4. Regenerate figures:
```bash
python tools/publication/generate_revised_figures.py
```
5. Regenerate manuscript package:
```bash
python tools/publication/generate_revised_manuscript.py
python tools/publication/generate_response_to_reviewers.py
```

## Paper-ready experiment matrix
Run at least these ablations:
- Base model only vs LoRA-tuned adapter.
- Prompt strategy: zero-shot vs few-shot vs chain-of-thought.
- Internal validation vs external hold-out set.
- Parse success and calibration quality.

Track metrics:
- Macro F1.
- Per-class recall (`high` class is safety critical).
- Parse rate.
- Over-classification rate to `high` risk.

## Risks and mitigations
- Small sample overfitting: use stratified splits and external hold-out.
- Label noise: dual clinician adjudication with conflict resolution.
- Hardware constraints: begin with small base model and LoRA adapters.
- Reproducibility drift: pin model IDs, seed values, and data snapshots.

## 30-day execution plan
- Week 1: dataset curation, annotation protocol, pilot training.
- Week 2: baseline and LoRA training runs, error analysis.
- Week 3: deployment hardening, latency/load profiling, governance checks.
- Week 4: manuscript update, figure finalization, submission package generation.
