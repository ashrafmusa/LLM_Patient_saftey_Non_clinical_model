"""Generate synthetic clinical scenarios for evaluation.

This module provides `generate_scenarios(n)` which returns a pandas DataFrame
with columns `id`, `text`, and `label` (low/medium/high). Labels are
assigned based on template content (useful for controlled model evaluation).
"""
import random
from typing import List
import pandas as pd

LOW_TEMPLATES: List[str] = [
    "Patient reports mild fatigue and occasional headache. No adverse events.",
    "Routine follow-up: vitals stable, no complaints.",
    "Patient denies chest pain, shortness of breath, or dizziness.",
]

MED_TEMPLATES: List[str] = [
    "Patient reports nausea and vomiting after new medication; possible intolerance.",
    "Minor medication dosing confusion occurred but was caught before administration.",
    "Patient experienced a fall without loss of consciousness; under observation.",
]

HIGH_TEMPLATES: List[str] = [
    "Medication error: ordered 10mg, administered 100mg leading to bradycardia.",
    "Suspected anaphylaxis after antibiotic administration with hypotension and rash.",
    "Patient expressed active suicidal ideation with plan and means.",
]

ALL_TEMPLATES = {"low": LOW_TEMPLATES, "medium": MED_TEMPLATES, "high": HIGH_TEMPLATES}


def generate_scenarios(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate `n` synthetic scenarios. Returns a DataFrame with `id`, `text`, `label`.

    The distribution will roughly balance across labels unless `n` is small.
    """
    random.seed(seed)
    rows = []
    labels = ["low", "medium", "high"]
    for i in range(1, n + 1):
        # sample label in round-robin-ish balanced way
        label = labels[(i - 1) % len(labels)]
        tmpl = random.choice(ALL_TEMPLATES[label])
        # add slight variation
        suffix = ""
        if random.random() < 0.5:
            suffix = " Patient history: " + random.choice(["hypertension", "diabetes", "no known allergies"])
        text = tmpl + suffix
        rows.append({"id": i, "text": text, "label": label})
    return pd.DataFrame(rows)
