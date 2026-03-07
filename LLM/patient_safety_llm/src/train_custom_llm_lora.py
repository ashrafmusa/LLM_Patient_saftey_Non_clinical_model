"""Train a project-specific instruction-tuned LLM adapter (LoRA) for patient safety risk tasks.

This script converts labeled clinical scenarios into instruction-response pairs and fine-tunes
an open-source base model with PEFT LoRA adapters.

Example:
    python -m src.train_custom_llm_lora \
      --data data/sample_clinical.csv \
      --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --output-dir models/custom_llm_lora
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

LABELS = {"low", "medium", "high"}


@dataclass
class Sample:
    prompt: str
    target: str


def _require_peft() -> tuple:
    """Import PEFT lazily so the script remains import-safe without optional deps."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PEFT is required. Install optional dependencies: pip install peft accelerate"
        ) from exc
    return LoraConfig, get_peft_model, prepare_model_for_kbit_training


def _build_instruction(text: str) -> str:
    return (
        "You are a clinical safety analyst. Classify the patient safety risk level for the scenario "
        "as exactly one of: low, medium, high. Return strict JSON with keys risk_level and reasoning.\n\n"
        f"Clinical scenario:\n{text.strip()}\n\n"
        "JSON response:"
    )


def _build_target(label: str, text: str) -> str:
    reasoning = {
        "low": "Routine or stable presentation with no immediate patient harm indicators.",
        "medium": "Moderate concern requiring monitoring or timely intervention.",
        "high": "Serious safety concern with potential immediate or severe patient harm.",
    }[label]
    payload = {
        "risk_level": label,
        "reasoning": reasoning,
        "evidence_excerpt": text[:180],
    }
    return json.dumps(payload, ensure_ascii=True)


def _load_samples(csv_path: Path, text_col: str, label_col: str) -> List[Sample]:
    df = pd.read_csv(csv_path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    samples: List[Sample] = []
    for _, row in df.iterrows():
        label = str(row[label_col]).strip().lower()
        text = str(row[text_col]).strip()
        if label not in LABELS or not text:
            continue
        samples.append(Sample(prompt=_build_instruction(text), target=_build_target(label, text)))

    if not samples:
        raise ValueError("No valid training rows found after filtering labels and empty text")
    return samples


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        merged = [f"{p}\n{t}{tokenizer.eos_token}" for p, t in zip(batch["prompt"], batch["target"])]
        encoded = tokenizer(
            merged,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(_tokenize, batched=True, remove_columns=list(dataset.features))


def _resolve_target_modules(model: AutoModelForCausalLM) -> List[str]:
    """Pick LoRA target modules that exist on the loaded model architecture."""
    module_names = [name for name, _ in model.named_modules()]
    candidate_sets = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA-style
        ["c_attn", "c_proj"],  # GPT-2 style
        ["query_key_value", "dense"],  # Some GPT-NeoX/BLOOM variants
    ]

    for candidates in candidate_sets:
        matched = [cand for cand in candidates if any(name.endswith(cand) for name in module_names)]
        if matched:
            return matched

    raise RuntimeError(
        "Could not infer LoRA target modules for this base model. "
        "Provide a model with known projection layer names (e.g., q_proj/c_attn)."
    )


def train_custom_lora(
    data: Path,
    output_dir: Path,
    base_model: str,
    text_col: str,
    label_col: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    use_4bit: bool,
) -> None:
    LoraConfig, get_peft_model, prepare_model_for_kbit_training = _require_peft()

    samples = _load_samples(data, text_col=text_col, label_col=label_col)
    ds = Dataset.from_list([{"prompt": s.prompt, "target": s.target} for s in samples])

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit mode requires bitsandbytes-compatible transformers install") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = _resolve_target_modules(model)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    tokenized = _tokenize_dataset(ds, tokenizer, max_length=max_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))

    metadata = {
        "base_model": base_model,
        "rows_used": len(samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "lora": {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "use_4bit": use_4bit,
        },
        "train_file": str(data),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom LoRA adapter for patient safety LLM tasks")
    parser.add_argument("--data", default="data/sample_clinical.csv")
    parser.add_argument("--output-dir", default="models/custom_llm_lora")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action="store_true")
    cli_args = parser.parse_args()

    train_custom_lora(
        data=Path(cli_args.data),
        output_dir=Path(cli_args.output_dir),
        base_model=cli_args.base_model,
        text_col=cli_args.text_col,
        label_col=cli_args.label_col,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        learning_rate=cli_args.learning_rate,
        max_length=cli_args.max_length,
        lora_rank=cli_args.lora_rank,
        lora_alpha=cli_args.lora_alpha,
        lora_dropout=cli_args.lora_dropout,
        use_4bit=cli_args.use_4bit,
    )
