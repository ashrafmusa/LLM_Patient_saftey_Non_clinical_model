"""Data augmentation and adversarial scenario generation.

Provides functions to expand template-driven scenarios via simple NLP
transformations (synonym replacement, negation flipping, typo injection)
and generate adversarial variants that alter clinical cues.
"""
import random
import re
from typing import List

SYNONYMS = {
    'severe': ['severe', 'intense', 'marked'],
    'mild': ['mild', 'slight', 'minor'],
    'error': ['error', 'mistake', 'incorrect order', 'ordering error'],
    'medication': ['medication', 'drug', 'medicine', 'dose'],
    'anaphylaxis': ['anaphylaxis', 'severe allergic reaction'],
    'suicidal': ['suicidal', 'self-harm', 'suicide ideation']
}


def synonym_replace(text: str, p: float = 0.2) -> str:
    """Randomly replace tokens with synonyms with probability `p`."""
    words = text.split()
    for i, w in enumerate(words):
        key = re.sub(r"[^a-zA-Z]", '', w).lower()
        if key in SYNONYMS and random.random() < p:
            choice = random.choice(SYNONYMS[key])
            # preserve punctuation
            suffix = ''
            if w and not w[-1].isalpha():
                suffix = w[-1]
            words[i] = choice + suffix
    return ' '.join(words)


def inject_typo(text: str, p: float = 0.05) -> str:
    """Randomly inject simple typos into the text."""
    def typo(word):
        if len(word) < 3:
            return word
        i = random.randrange(len(word)-1)
        lst = list(word)
        lst[i], lst[i+1] = lst[i+1], lst[i]
        return ''.join(lst)

    words = text.split()
    for i, w in enumerate(words):
        if random.random() < p:
            # skip punctuation-only
            if re.match(r"^\W+$", w):
                continue
            words[i] = typo(w)
    return ' '.join(words)


def negate_statement(text: str) -> str:
    """Simple negation flip for certain phrases (adversarial)."""
    # flip "no" to "yes" and vice versa for simple patterns
    text = re.sub(r"\bno known allergies\b", "has known allergies", text, flags=re.I)
    text = re.sub(r"\bno complaints\b", "new complaints", text, flags=re.I)
    return text


def augment_row(text: str, n_variants: int = 3) -> List[str]:
    variants = []
    for _ in range(n_variants):
        t = text
        if random.random() < 0.6:
            t = synonym_replace(t, p=0.25)
        if random.random() < 0.3:
            t = inject_typo(t, p=0.05)
        if random.random() < 0.2:
            t = negate_statement(t)
        variants.append(t)
    return variants


def augment_dataframe(df, text_column: str = 'text', multiplier: int = 3, seed: int = 42):
    """Return a new dataframe with augmented scenarios appended.

    Each original row will produce `multiplier` augmented variants.
    """
    random.seed(seed)
    rows = []
    for _, row in df.iterrows():
        rows.append(row.to_dict())
        orig = str(row.get(text_column, ''))
        for v in augment_row(orig, n_variants=multiplier):
            new = dict(row)
            new[text_column] = v
            rows.append(new)
    # convert to pandas DataFrame lazily to avoid adding pandas dependency here
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows
