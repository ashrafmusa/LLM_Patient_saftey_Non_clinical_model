"""Cross-validated training with detailed reporting and figure generation.

Performs StratifiedKFold CV, records per-fold reports, and aggregates calibration
and ROC plots across folds. Saves `cv_report.json` and figures to `reports/`.
"""
import os
import json
import joblib
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

from .data_ingest import load_csv, preprocess_dataframe
from .augment import augment_dataframe
from .plots import generate_calibration_and_roc


def train_cv(input_csv: str, n_splits: int = 5, augment_multiplier: int = 2):
    df = load_csv(input_csv, text_column='text')
    # augment data
    try:
        aug_df = augment_dataframe(df, text_column='text', multiplier=augment_multiplier)
    except Exception:
        aug_df = df

    pdf = preprocess_dataframe(aug_df, text_column='text')
    # create labels (heuristic if needed)
    if 'label' in df.columns:
        labels = pdf['label'].map({'low':0,'medium':1,'high':2}).fillna(0).astype(int)
    else:
        from .train import heuristic_label
        labels = pdf['processed_text'].apply(heuristic_label)

    X = pdf['processed_text'].values
    y = labels.values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_reports = []
    fold_idx = 0
    all_preds = []

    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fit vectorizer on fold train only to avoid leakage
        vect_fold = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))
        Xt = vect_fold.fit_transform(X_train)
        clf = LogisticRegression(max_iter=500, class_weight='balanced', solver='saga')
        clf.fit(Xt, y_train)
        # calibrate
        calib = CalibratedClassifierCV(clf, cv='prefit', method='sigmoid')
        calib.fit(Xt, y_train)

        X_test_t = vect_fold.transform(X_test)
        preds = calib.predict(X_test_t)
        probs = calib.predict_proba(X_test_t)

        report = classification_report(y_test, preds, output_dict=True)
        # compute multiclass AUC using one-vs-rest
        try:
            auc = roc_auc_score(pd.get_dummies(y_test), probs)
        except Exception:
            auc = None

        fold_reports.append({'fold': fold_idx, 'report': report, 'auc': auc})

        # collect per-sample results for aggregated evaluation
        for i, idx in enumerate(test_idx):
            all_preds.append({
                'id': int(pdf.iloc[idx].get('id', idx)),
                'orig_text': pdf.iloc[idx]['orig_text'],
                'processed_text': pdf.iloc[idx]['processed_text'],
                'truth': ['low', 'medium', 'high'][int(y_test[i])],
                'predicted': ['low', 'medium', 'high'][int(preds[i])],
                'score': float(probs[i].max()),
                'model_based': True,
                'explain_available': False,
                'explain': None,
            })

    # save last calibrated model and last-fold vectorizer for convenience
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(calib, os.path.join(models_dir, 'risk_model.pkl'))
    try:
        joblib.dump(vect_fold, os.path.join(models_dir, 'vectorizer.pkl'))
    except Exception:
        pass

    # save CV report
    now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = os.path.abspath(os.path.join('reports', f'cv_report_{now}'))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'cv_report.json'), 'w') as f:
        json.dump(fold_reports, f, indent=2)

    # write aggregated evaluation CSV
    results_df = pd.DataFrame(all_preds)
    eval_csv = os.path.join(out_dir, 'cv_aggregated_evaluation.csv')
    results_df.to_csv(eval_csv, index=False)

    # compute simple metrics
    metrics = {}
    if 'truth' in results_df.columns and results_df['truth'].notna().any():
        metrics['total'] = len(results_df)
        metrics['accuracy'] = float((results_df['truth'] == results_df['predicted']).mean())
        metrics['by_label'] = {}
        for lbl in ['low', 'medium', 'high']:
            sub = results_df[results_df['truth'] == lbl]
            metrics['by_label'][lbl] = {
                'count': int(len(sub)),
                'accuracy': float((sub['truth'] == sub['predicted']).mean() if len(sub) > 0 else 0.0),
            }
    # save metrics
    with open(os.path.join(out_dir, 'cv_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # generate calibration and ROC plots
    try:
        plot_paths = generate_calibration_and_roc(eval_csv, out_dir=os.path.join(out_dir, 'figures'))
    except Exception as e:
        plot_paths = {'error': str(e)}

    return {'cv_report': os.path.join(out_dir, 'cv_report.json'), 'models_dir': models_dir, 'eval_csv': eval_csv, 'metrics': metrics, 'plots': plot_paths}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--augment', type=int, default=2)
    args = parser.parse_args()
    r = train_cv(args.input, n_splits=args.n_splits, augment_multiplier=args.augment)
    print(r)
