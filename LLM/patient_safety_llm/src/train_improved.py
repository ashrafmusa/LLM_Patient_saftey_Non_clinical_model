"""Improved training pipeline: class rebalancing, hyperparameter tuning, and calibration.

Saves calibrated classifier and TF-IDF vectorizer to `models/`.
"""
import os
import joblib
import logging
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from .data_ingest import load_csv, preprocess_dataframe

logger = logging.getLogger(__name__)


def train_improved(input_csv: str, text_col: str = 'text', label_col: Optional[str] = 'label', extra_names_col: Optional[str] = None):
    df = load_csv(input_csv, text_column=text_col)
    pdf = preprocess_dataframe(df, text_column=text_col, extra_names_column=extra_names_col)

    if label_col and label_col in df.columns:
        labels = df[label_col].map({'low': 0, 'medium': 1, 'high': 2}).fillna(0).astype(int)
    else:
        # fallback to heuristic labels
        from .train import heuristic_label
        labels = pdf['processed_text'].apply(heuristic_label)

    X = pdf['processed_text']
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    tfidf = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1,2))
    X_train_t = tfidf.fit_transform(X_train)

    # Logistic regression with class_weight balanced
    base_clf = LogisticRegression(max_iter=500, solver='saga', class_weight='balanced')

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base_clf, param_grid, cv=cv, scoring='f1_macro', n_jobs=1)
    gs.fit(X_train_t, y_train)

    best = gs.best_estimator_

    # Calibrate probabilities
    calib = CalibratedClassifierCV(best, cv='prefit', method='sigmoid')
    calib.fit(X_train_t, y_train)

    # Evaluate on test
    X_test_t = tfidf.transform(X_test)
    preds = calib.predict(X_test_t)
    report = classification_report(y_test, preds, output_dict=True)
    print(classification_report(y_test, preds))

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'risk_model.pkl')
    vect_path = os.path.join(models_dir, 'vectorizer.pkl')

    joblib.dump(calib, model_path)
    joblib.dump(tfidf, vect_path)

    return {'model_path': model_path, 'vect_path': vect_path, 'report': report}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--text_col', default='text')
    parser.add_argument('--label_col', default='label')
    parser.add_argument('--extra_names_col', default=None)
    args = parser.parse_args()
    res = train_improved(args.input, text_col=args.text_col, label_col=args.label_col, extra_names_col=args.extra_names_col)
    print(res)
