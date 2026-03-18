"""Run first project aim with real LLM inference on labeled clinical scenarios.

Usage:
  python run_real_llm_study.py --data data/sample_clinical.csv

Environment (OpenAI-compatible path):
    OPENAI_API_KEY=your_api_key_here
  OPENAI_BASE_URL=https://api.openai.com/v1
  OPENAI_MODEL=gpt-4o-mini

Alternative local path:
  LLAMA_SERVER_URL=http://127.0.0.1:8080

One-command provider presets:
    python run_real_llm_study.py --provider grok --data data/sample_clinical.csv
    python run_real_llm_study.py --provider ollama --data data/sample_clinical.csv
    python run_real_llm_study.py --provider llama_cpp --data data/sample_clinical.csv
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.llm_evaluation import evaluate_llm_on_scenarios
from src.config import settings


def _load_scenarios(data_path: Path) -> list[dict]:
    df = pd.read_csv(data_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep only valid labels for study consistency.
    valid = df[df["label"].isin(["low", "medium", "high"])].copy()
    return valid[["text", "label"]].to_dict(orient="records")


def _set_openai_compatible(base_url: str, model: str, api_key: str | None) -> None:
    # Update both process env and in-memory settings so imported modules see the change.
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_MODEL"] = model
    settings.OPENAI_BASE_URL = base_url
    settings.OPENAI_MODEL = model
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        settings.OPENAI_API_KEY = api_key


def _set_llama_server(url: str) -> None:
    os.environ["LLAMA_SERVER_URL"] = url
    settings.LLAMA_SERVER_URL = url


def _apply_provider_preset(
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    llama_server_url: str | None,
) -> str:
    """Apply provider preset while preserving default legacy behavior on `auto`."""
    provider = provider.lower()

    if provider == "auto":
        return "auto(env-based legacy configuration)"

    if provider == "grok":
        _set_openai_compatible(
            base_url=base_url or "https://api.x.ai/v1",
            model=model or "grok-2-latest",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        return f"grok(model={settings.OPENAI_MODEL}, base={settings.OPENAI_BASE_URL})"

    if provider == "openai":
        _set_openai_compatible(
            base_url=base_url or "https://api.openai.com/v1",
            model=model or "gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        return f"openai(model={settings.OPENAI_MODEL}, base={settings.OPENAI_BASE_URL})"

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint at /v1.
        _set_openai_compatible(
            base_url=base_url or "http://127.0.0.1:11434/v1",
            model=model or "llama3.1:8b",
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "ollama",
        )
        return f"ollama(model={settings.OPENAI_MODEL}, base={settings.OPENAI_BASE_URL})"

    if provider == "llama_cpp":
        _set_llama_server(llama_server_url or "http://127.0.0.1:8080")
        if model:
            # Optional and only meaningful if your server routes by model name.
            settings.OPENAI_MODEL = model
        return f"llama_cpp(server={settings.LLAMA_SERVER_URL})"

    raise ValueError(f"Unsupported provider: {provider}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-LLM risk classification study")
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "grok", "openai", "ollama", "llama_cpp"],
        help="Provider preset. 'auto' keeps original env-based behavior.",
    )
    parser.add_argument(
        "--data",
        default="data/sample_clinical.csv",
        help="Path to CSV with at least columns: text,label",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the selected provider.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional base URL override for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. If omitted, existing env variables are used.",
    )
    parser.add_argument(
        "--llama-server-url",
        default=None,
        help="Optional llama.cpp server URL override (for --provider llama_cpp).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero_shot", "few_shot", "chain_of_thought", "safety_focused"],
        choices=["zero_shot", "few_shot", "chain_of_thought", "safety_focused"],
        help="Prompting strategies to evaluate",
    )
    args = parser.parse_args()

    provider_info = _apply_provider_preset(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        llama_server_url=args.llama_server_url,
    )
    print(f"Provider configuration: {provider_info}")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    scenarios = _load_scenarios(data_path)
    if not scenarios:
        raise ValueError("No valid scenarios found after filtering labels")

    result = evaluate_llm_on_scenarios(
        scenarios=scenarios,
        strategies=args.strategies,
        use_simulation=False,
    )

    # Hard guard: we only accept real parsed outputs for this run mode.
    parsed_total = sum(1 for r in result["results"] if r.get("risk_level") is not None)
    if parsed_total == 0:
        raise RuntimeError(
            "No parseable LLM outputs found. Check OPENAI_* or LLAMA_* configuration, then retry."
        )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("reports") / f"real_llm_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    results_path = out_dir / "results.csv"

    summary_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    pd.DataFrame(result["results"]).to_csv(results_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved results: {results_path}")
    print("Per-strategy accuracy:")
    for strategy, stats in result["summary"].items():
        acc = stats.get("accuracy", 0.0)
        parse_rate = stats.get("parse_rate", 0.0)
        print(f"  - {strategy}: accuracy={acc:.3f}, parse_rate={parse_rate:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
