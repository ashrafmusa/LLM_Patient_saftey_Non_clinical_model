"""Data ingestion and preprocessing utilities for the patient safety LLM project.

Provides CSV loading, basic cleaning, de-identification integration, and train/test split helper.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from .deid import deidentify_text


def load_csv(path: str, text_column: str = "text") -> pd.DataFrame:
    """Load a CSV and ensure the text column exists."""
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in CSV")
    return df


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace and strip."""
    if text is None:
        return ""
    out = str(text).strip()
    out = " ".join(out.split())
    return out


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text", extra_names_column: str = None) -> pd.DataFrame:
    """Return a copy of `df` with cleaned and de-identified text in `processed_text` column.

    The function uses `deidentify_text` to redact PHI; it keeps the original text in `orig_text`.
    """
    out = df.copy()
    processed = []
    found = []
    for _, row in out.iterrows():
        txt = clean_text(row.get(text_column, ""))
        extra = None
        if extra_names_column and extra_names_column in out.columns:
            val = row.get(extra_names_column)
            extra = [v.strip() for v in str(val).split(';') if v]
        res = deidentify_text(txt, extra_names=extra)
        processed.append(res["text"])
        found.append(res["found"])
    out["orig_text"] = out[text_column]
    out["processed_text"] = processed
    out["redaction_summary"] = found
    return out


def split_dataset(df: pd.DataFrame, text_column: str = "processed_text", test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/test by rows."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)
