# Aim 1 Plan: Real LLM Risk Classification Study

## Aim
Evaluate patient safety risk classification on clinical scenarios using real LLM inference (not simulated outputs).

## Scope
- Input data: CSV with `text,label` columns (`low`, `medium`, `high`).
- Prompting strategies: `zero_shot`, `few_shot`, `chain_of_thought`.
- Outputs: strategy-level summary metrics + row-level prediction records.

## Run Command
```bash
python run_real_llm_study.py --data data/sample_clinical.csv
```

## Required Environment
OpenAI-compatible route:
```bash
set OPENAI_API_KEY=your_key
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
```

Local llama route (optional):
```bash
set LLAMA_SERVER_URL=http://127.0.0.1:8080
```

## Success Criteria
- Parse rate >= 0.95 for each strategy.
- Non-trivial confusion matrix (not one-class collapse across all rows).
- Reproducible artifacts saved under `reports/real_llm_eval_<timestamp>/`.

## Notes
- This run mode sets `use_simulation=False` and fails fast if no parseable real outputs are returned.
- Keep synthetic pipeline only for baseline comparison and methodology checks.
