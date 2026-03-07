"""Serve a custom fine-tuned LLM adapter via FastAPI for project-specific inference.

Example:
    uvicorn src.serve_custom_llm_api:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class PredictRequest(BaseModel):
    text: str
    max_new_tokens: int = 180


class PredictResponse(BaseModel):
    risk_level: str
    reasoning: str
    raw_response: str


def _build_prompt(text: str) -> str:
    return (
        "You are a clinical safety analyst. Classify the patient safety risk level for the scenario "
        "as exactly one of low, medium, high. Return strict JSON with keys risk_level and reasoning.\n\n"
        f"Clinical scenario:\n{text.strip()}\n\n"
        "JSON response:"
    )


def _safe_parse_risk(raw: str) -> Dict[str, str]:
    try:
        match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group())
            risk_level = str(payload.get("risk_level", "unknown")).lower().strip()
            reasoning = str(payload.get("reasoning", "")).strip()
            if risk_level in {"low", "medium", "high"}:
                return {"risk_level": risk_level, "reasoning": reasoning}
    except Exception:
        pass

    lowered = raw.lower()
    for level in ("high", "medium", "low"):
        if level in lowered:
            return {"risk_level": level, "reasoning": raw[:300]}

    return {"risk_level": "unknown", "reasoning": raw[:300]}


@lru_cache(maxsize=1)
def _load_model() -> tuple:
    base_model = os.getenv("CUSTOM_LLM_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_path = os.getenv("CUSTOM_LLM_ADAPTER_PATH", "models/custom_llm_lora/adapter")
    adapter_exists = os.path.isdir(adapter_path)

    tokenizer_source = adapter_path if adapter_exists else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Optional PEFT adapter merge when artifacts are present.
    if adapter_exists:
        try:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass

    model.eval()
    return tokenizer, model


app = FastAPI(title="Custom Patient Safety LLM API")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    tokenizer, model = _load_model()
    prompt = _build_prompt(req.text)

    encoded = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    completion = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded
    parsed = _safe_parse_risk(completion)

    return PredictResponse(
        risk_level=parsed["risk_level"],
        reasoning=parsed["reasoning"],
        raw_response=completion[:1200],
    )
