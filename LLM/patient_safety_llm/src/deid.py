"""De-identification utilities for clinical text.

This module provides lightweight, best-effort PHI redaction suitable for
prototype pipelines. It is NOT a production-grade de-id solution. Replace or
augment with specialized libraries for real PHI.
"""
import re
from typing import Tuple, List, Dict

import pandas as pd


# Regular expressions for common PHI patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{2,4}\)|\d{2,4})[-.\s]?\d{2,4}[-.\s]?\d{2,6}\b")
_MRN_RE = re.compile(r"\b(?:MRN|Medical Record Number|Record#)[:#\s]*([0-9A-Za-z-]{4,20})\b", re.I)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_DATE_RE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*\d{2,4})\b", re.I)
_ID_RE = re.compile(r"\b(ID|id)[:\s]*[A-Za-z0-9-]{3,}\b")
_AGE_RE = re.compile(r"\b(\d{1,3})\s*-?year[s]?\b", re.I)


def deidentify_text(text: str, extra_names: List[str] = None) -> Dict[str, object]:
    """Return a de-identified version of `text` and a dict of findings.

    This function redacts emails, phones, dates, SSNs, MRNs, simple IDs,
    and ages. `extra_names` can include patient/staff names to be redacted.
    """
    if text is None:
        return {"text": "", "found": {}}

    found = {"emails": [], "phones": [], "dates": [], "ssns": [], "mrns": [], "ids": [], "ages": [], "names": []}
    out = text

    # Emails
    for m in _EMAIL_RE.findall(out):
        found["emails"].append(m)
    out = _EMAIL_RE.sub("[EMAIL]", out)

    # Phones
    for m in _PHONE_RE.findall(out):
        # regex includes groups sometimes; ensure string
        found["phones"].append(m if isinstance(m, str) else ''.join(m))
    out = _PHONE_RE.sub("[PHONE]", out)

    # SSN
    for m in _SSN_RE.findall(out):
        found["ssns"].append(m)
    out = _SSN_RE.sub("[SSN]", out)

    # MRN / record tokens
    for m in _MRN_RE.findall(out):
        found["mrns"].append(m)
    out = _MRN_RE.sub("[MRN]", out)

    # Generic ID tokens
    for m in _ID_RE.findall(out):
        found["ids"].append(m)
    out = _ID_RE.sub("[ID]", out)

    # Dates
    for m in _DATE_RE.findall(out):
        found["dates"].append(m)
    out = _DATE_RE.sub("[DATE]", out)

    # Ages (e.g., 45-year-old)
    for m in _AGE_RE.findall(out):
        found["ages"].append(m)
    out = _AGE_RE.sub("[AGE]", out)

    # Extra name redaction (best-effort exact matches)
    if extra_names:
        for nm in extra_names:
            if not nm:
                continue
            esc = re.escape(nm)
            names_re = re.compile(r"\b" + esc + r"\b", re.I)
            if names_re.search(out):
                found["names"].append(nm)
            out = names_re.sub("[NAME]", out)

    return {"text": out, "found": found}


def deidentify_csv(input_csv: str, output_csv: str, text_column: str = "text", extra_names_column: str = None) -> Dict:
    """Load a CSV, de-identify a text column, and write result to `output_csv`.

    Returns a summary dict with counts and sample findings.
    """
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in CSV")

    results = []
    summary = {"rows": len(df), "redactions": 0}
    for _, row in df.iterrows():
        extra = None
        if extra_names_column and extra_names_column in df.columns:
            val = row.get(extra_names_column)
            extra = [v.strip() for v in str(val).split(';') if v]
        res = deidentify_text(str(row[text_column]), extra_names=extra)
        results.append(res["text"])
        # count any found items
        if any(res["found"].values()):
            summary["redactions"] += 1

    out_df = df.copy()
    out_df[text_column] = results
    out_df.to_csv(output_csv, index=False)
    return summary
