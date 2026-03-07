Project plan

1. Define scope and data sources (EHR, simulated cases, clinical guidelines).
2. Data ingestion and de-identification pipeline.
3. LLM integration (choose local `llama.cpp` or remote APIs).
4. Implement risk-stratified assessment module (clinical safety rules + LLM scoring).
5. Evaluation: correctness, hallucination, harmful recommendations, latency, and privacy.

Model choices and tradeoffs
- Local (llama.cpp): good for privacy/control, heavier setup.
- Remote (OpenAI/Azure): faster to prototype, but needs PHI handling and compliance.

Next actions
- Confirm model preference and available datasets.
