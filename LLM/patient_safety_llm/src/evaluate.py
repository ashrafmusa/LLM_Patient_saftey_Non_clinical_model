"""Evaluation harness: run scenarios through the pipeline, collect metrics and reports.

Usage:
    python -m src.evaluate --n 200 --output reports/eval_200

If no trained model exists in `models/`, the script will train one using the
generated scenarios (heuristic labels) as a bootstrap.
"""
import os
import json
import argparse
from datetime import datetime

import pandas as pd
import joblib

from .generate_scenarios import generate_scenarios
from .data_ingest import preprocess_dataframe
from .train import train
from .risk_assessment import assess_risk
from .explain import explain_text


def ensure_model(scen_df, work_dir):
    """Ensure a model exists; if not, train using `scen_df` saved to a CSV."""
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'risk_model.pkl')
    vect_path = os.path.join(models_dir, 'vectorizer.pkl')
    if os.path.exists(model_path) and os.path.exists(vect_path):
        return

    tmp_csv = os.path.join(work_dir, 'scenarios_for_training.csv')
    scen_df.to_csv(tmp_csv, index=False)
    args = type('A', (), {'input': tmp_csv, 'text_col': 'text', 'label_col': 'label', 'extra_names_col': None})
    train(args)


def evaluate(n: int = 200, output_dir: str = 'reports/eval') -> dict:
    now = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = os.path.abspath(output_dir + '_' + now)
    os.makedirs(out_dir, exist_ok=True)

    df = generate_scenarios(n)
    ensure_model(df, out_dir)

    # Preprocess (de-identify) scenarios
    pdf = preprocess_dataframe(df, text_column='text')

    results = []
    for _, row in pdf.iterrows():
        text = row['processed_text']
        truth = row.get('label')
        res = assess_risk(text)
        expl = explain_text(text)
        results.append({
            'id': row['id'],
            'orig_text': row['orig_text'],
            'processed_text': text,
            'truth': truth,
            'predicted': res.get('risk_level'),
            'score': res.get('score'),
            'model_based': res.get('model_based', False),
            'explain_available': expl.get('available', False),
            'explain': expl.get('explanations') or None
        })

    out_df = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, 'evaluation_results.csv')
    out_df.to_csv(out_csv, index=False)

    # Simple accuracy metric (truth vs predicted)
    metrics = {}
    if 'truth' in out_df.columns and out_df['truth'].notna().any():
        metrics['total'] = len(out_df)
        metrics['accuracy'] = float((out_df['truth'] == out_df['predicted']).mean())
        metrics['by_label'] = {}
        for lbl in out_df['truth'].unique():
            sub = out_df[out_df['truth'] == lbl]
            metrics['by_label'][lbl] = {
                'count': int(len(sub)),
                'accuracy': float((sub['truth'] == sub['predicted']).mean())
            }

    metrics['explainability_coverage'] = float(out_df['explain_available'].mean())

    metrics_path = os.path.join(out_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'out_dir': out_dir, 'results_csv': out_csv, 'metrics': metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200)
    parser.add_argument('--output', default='reports/eval')
    args = parser.parse_args()
    r = evaluate(n=args.n, output_dir=args.output)
    print('Evaluation complete. Results in', r['out_dir'])
