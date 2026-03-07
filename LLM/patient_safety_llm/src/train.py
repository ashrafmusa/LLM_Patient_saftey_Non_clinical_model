"""Training utilities for a lightweight risk classifier.

This script supports bootstrapped training using heuristic labels when no
manually labeled dataset is available. It saves a sklearn pipeline (vectorizer
and classifier) into `models/`.

Usage example:
    python -m src.train --input data/sample_clinical.csv --text_col text --label_col label
"""
import os
import argparse
import joblib
import logging

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

from .data_ingest import load_csv, preprocess_dataframe
from .risk_assessment import _heuristic_score

logger = logging.getLogger(__name__)


def heuristic_label(text: str) -> int:
    """Create a pseudo-label from heuristic: 0=low,1=medium,2=high"""
    res = _heuristic_score(text)
    lvl = res.get('risk_level', 'low')
    return {'low': 0, 'medium': 1, 'high': 2}.get(lvl, 0)


def train(args):
    df = load_csv(args.input, text_column=args.text_col)
    pdf = preprocess_dataframe(df, text_column=args.text_col, extra_names_column=args.extra_names_col)

    # Determine labels: prefer explicit label column, else heuristic labels
    if args.label_col and args.label_col in df.columns:
        labels = df[args.label_col].map({'low': 0, 'medium': 1, 'high': 2}).fillna(0).astype(int)
    else:
        labels = pdf['processed_text'].apply(heuristic_label)

    X_train, X_test, y_train, y_test = train_test_split(pdf['processed_text'], labels, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1,2))),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # If training labels contain only a single class (small datasets / heuristic labels),
    # fall back to a DummyClassifier to avoid solver errors.
    unique_labels = set(y_train.tolist()) if hasattr(y_train, 'tolist') else set(y_train)
    if len(unique_labels) < 2:
        clf = DummyClassifier(strategy='most_frequent')
        # fit vectorizer then classifier
        tfidf = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1,2))
        Xt = tfidf.fit_transform(X_train)
        clf.fit(Xt, y_train)
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('clf', clf)
        ])
    else:
        pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.abspath(os.path.join(out_dir, 'risk_model.pkl'))
    vect_path = os.path.abspath(os.path.join(out_dir, 'vectorizer.pkl'))

    # Save model and vectorizer separately for compatibility with risk_assessment loader
    joblib.dump(pipeline.named_steps['clf'], model_path)
    joblib.dump(pipeline.named_steps['tfidf'], vect_path)
    print(f"Saved model to {model_path} and vectorizer to {vect_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--text_col', default='text')
    parser.add_argument('--label_col', default=None)
    parser.add_argument('--extra_names_col', default=None)
    args = parser.parse_args()
    train(args)
