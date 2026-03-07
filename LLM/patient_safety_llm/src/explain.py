"""Explainability helpers for the risk classifier.

This module provides a lightweight, dependency-light explanation approach
for linear models: it computes per-feature contributions using model
coefficients and a TF-IDF vectorizer. If a non-linear model is present
and SHAP is installed, a richer SHAP explanation can be attempted.
"""
import os
import logging
from typing import Dict, List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_model.pkl')
_VECT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.pkl')


def _load_model_and_vectorizer():
    try:
        model = joblib.load(os.path.abspath(_MODEL_PATH))
        vect = joblib.load(os.path.abspath(_VECT_PATH))
        return model, vect
    except Exception as e:
        logger.debug("No model/vectorizer for explanation: %s", e)
        return None, None


def explain_text(text: str, top_k: int = 10) -> Dict:
    """Return a lightweight explanation for `text`.

    For linear classifiers (e.g., LogisticRegression), compute the per-feature
    contribution: coef * tfidf_value. Return the top_k features by absolute
    contribution. If no model is available, return availability=False.
    """
    model, vect = _load_model_and_vectorizer()
    if not model or not vect:
        return {"available": False, "reason": "no_model"}

    try:
        X = vect.transform([text])
    except Exception as e:
        return {"available": False, "reason": f"vectorize_failed: {e}"}

    # Prefer linear coefficient inspection
    if hasattr(model, 'coef_'):
        try:
            # For multi-class, pick predicted class
            if hasattr(model, 'predict'):
                pred = int(model.predict(X)[0])
            else:
                pred = 0

            coefs = model.coef_
            # coefs shape: (n_classes, n_features) or (n_features,)
            if coefs.ndim == 1:
                coef_vec = coefs
            else:
                coef_vec = coefs[pred]

            feat_names = vect.get_feature_names_out()
            x_arr = X.toarray()[0]
            contributions = coef_vec * x_arr

            idx = np.argsort(np.abs(contributions))[::-1][:top_k]
            explanations: List[Dict] = []
            for i in idx:
                if x_arr[i] == 0 and contributions[i] == 0:
                    continue
                explanations.append({
                    "feature": feat_names[i],
                    "tfidf": float(x_arr[i]),
                    "coef": float(coef_vec[i]),
                    "contribution": float(contributions[i])
                })

            return {"available": True, "model_based": True, "predicted_class": pred, "explanations": explanations}
        except Exception as e:
            logger.debug("Linear explanation failed: %s", e)
            # fall through to try SHAP if available

    # Try SHAP if installed
    try:
        import shap
        # define a predict_fn that accepts list[str]
        def predict_fn(texts):
            return model.predict_proba(vect.transform(texts))

        # use a small background (empty string)
        background = [""]
        explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1], shap.maskers.Text(vect))
        shap_values = explainer.shap_values([text])
        return {"available": True, "model_based": True, "shap_values": shap_values}
    except Exception as e:
        logger.debug("SHAP explanation not available or failed: %s", e)

    return {"available": False, "reason": "no_explanation_available"}
