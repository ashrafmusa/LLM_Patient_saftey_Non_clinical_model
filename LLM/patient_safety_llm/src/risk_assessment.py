"""
Risk-stratified assessment utilities.
"""
from typing import Dict
import os
import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_model.pkl')
_VECT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')


def _heuristic_score(text: str) -> Dict:
    """Original heuristic fallback scoring."""
    lowered = (text or '').lower()
    score = 0
    if 'suicide' in lowered or 'self-harm' in lowered:
        score += 5
    if 'error' in lowered or 'mistak' in lowered:
        score += 3
    if 'allergy' in lowered or 'reaction' in lowered:
        score += 4

    if score >= 5:
        level = 'high'
    elif score >= 3:
        level = 'medium'
    else:
        level = 'low'

    return {"risk_level": level, "score": score, "matched_terms": []}


def _load_model():
    """Try to load a trained sklearn model and vectorizer. Returns (model, vectorizer) or (None, None)."""
    try:
        model = joblib.load(os.path.abspath(_MODEL_PATH))
        vect = joblib.load(os.path.abspath(_VECT_PATH))
        return model, vect
    except Exception as e:
        logger.debug("Model or vectorizer not found or failed to load: %s", e)
        return None, None


def assess_risk(text: str) -> Dict:
    """Assess risk using a trained model when available, otherwise heuristic fallback.

    Model should output a probability for an adverse event; we map probabilities
    to `low`/`medium`/`high` using thresholds.
    """
    model, vect = _load_model()
    if model and vect:
        try:
            X = vect.transform([text or ''])
            # assume model has predict_proba
            if hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba(X)[0].max())
            else:
                # fallback to decision_function mapping
                score = float(model.decision_function(X)[0])
                prob = 1.0 / (1.0 + pow(2.718281828, -score))

            if prob >= 0.75:
                level = 'high'
            elif prob >= 0.45:
                level = 'medium'
            else:
                level = 'low'

            return {"risk_level": level, "score": prob, "model_based": True}
        except Exception as e:
            logger.debug("Model prediction failed, falling back to heuristic: %s", e)

    # fallback
    return _heuristic_score(text)
