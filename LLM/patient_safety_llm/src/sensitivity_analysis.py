"""
Sensitivity and robustness analyses for risk classification pipeline.

Addresses reviewer concerns about:
- Augmentation strategy sensitivity
- Baseline/alternative model comparison
- Class imbalance robustness
- Domain shift evaluation
- Threshold calibration and cost-sensitive evaluation
"""

import logging
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from .augment import inject_typo, negate_statement as negate, synonym_replace

logger = logging.getLogger(__name__)


# ── Augmentation strategy sensitivity ───────────────────────────────────────

def augmentation_sensitivity_analysis(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    n_splits: int = 5,
) -> Dict:
    """Evaluate model performance under different augmentation strategies.

    Compares: no augmentation, synonym-only, typo-only, negation-only,
    all combined, and varying augmentation multipliers.

    Returns
    -------
    dict mapping strategy name → cross-validated metrics
    """
    strategies = {
        "no_augmentation": [],
        "synonym_only": [synonym_replace],
        "typo_only": [inject_typo],
        "negation_only": [negate],
        "synonym+typo": [synonym_replace, inject_typo],
        "all_combined": [synonym_replace, inject_typo, negate],
    }

    results = {}

    for strategy_name, augment_fns in strategies.items():
        logger.info("Evaluating augmentation strategy: %s", strategy_name)

        # Build augmented dataset
        aug_texts = list(scenarios[text_col])
        aug_labels = list(scenarios[label_col])

        if augment_fns:
            for _, row in scenarios.iterrows():
                for fn in augment_fns:
                    aug_text = fn(row[text_col])
                    aug_texts.append(aug_text)
                    aug_labels.append(row[label_col])

        # Cross-validate
        metrics = _cross_validate(aug_texts, aug_labels, n_splits=n_splits)
        metrics["n_samples"] = len(aug_texts)
        metrics["augmentation_ratio"] = len(aug_texts) / len(scenarios)
        results[strategy_name] = metrics

    return results


def augmentation_multiplier_analysis(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    multipliers: Optional[List[int]] = None,
    n_splits: int = 5,
) -> Dict:
    """Test effect of varying augmentation multiplier (1x, 2x, 3x, 5x, 10x)."""
    if multipliers is None:
        multipliers = [1, 2, 3, 5, 10]

    augment_fns = [synonym_replace, inject_typo, negate]
    results = {}

    for mult in multipliers:
        aug_texts = list(scenarios[text_col])
        aug_labels = list(scenarios[label_col])

        for _ in range(mult - 1):
            for _, row in scenarios.iterrows():
                fn = random.choice(augment_fns)
                aug_texts.append(fn(row[text_col]))
                aug_labels.append(row[label_col])

        metrics = _cross_validate(aug_texts, aug_labels, n_splits=n_splits)
        metrics["multiplier"] = mult
        metrics["n_samples"] = len(aug_texts)
        results[f"{mult}x"] = metrics

    return results


# ── Alternative model comparison ────────────────────────────────────────────

def model_comparison_analysis(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    augment: bool = True,
    n_splits: int = 5,
) -> Dict:
    """Compare multiple classification approaches as baselines.

    Models: Logistic Regression, Multinomial Naive Bayes, Linear SVM,
    Random Forest, Gradient Boosting, Majority-class Dummy.
    """
    texts = list(scenarios[text_col])
    labels = list(scenarios[label_col])

    if augment:
        for _, row in scenarios.iterrows():
            for fn in [synonym_replace, inject_typo, negate]:
                texts.append(fn(row[text_col]))
                labels.append(row[label_col])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0
        ),
        "Multinomial NB": MultinomialNB(alpha=1.0),
        "Linear SVM": LinearSVC(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "Majority Class (Dummy)": DummyClassifier(strategy="most_frequent"),
    }

    results = {}
    for model_name, model in models.items():
        logger.info("Evaluating model: %s", model_name)
        metrics = _cross_validate(
            texts, labels, n_splits=n_splits, model=model
        )
        results[model_name] = metrics

    return results


# ── Class imbalance robustness ──────────────────────────────────────────────

def imbalance_robustness_analysis(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    n_splits: int = 5,
) -> Dict:
    """Evaluate model under realistic class imbalance conditions.

    Tests: balanced (1:1:1), moderate imbalance (2:1:1),
    severe imbalance (5:2:1), extreme (10:3:1).
    """
    imbalance_configs = {
        "balanced_1:1:1": {"low": 1.0, "medium": 1.0, "high": 1.0},
        "moderate_2:1:1": {"low": 2.0, "medium": 1.0, "high": 1.0},
        "clinical_1:2:5": {"low": 1.0, "medium": 2.0, "high": 5.0},
        "severe_5:2:1": {"low": 5.0, "medium": 2.0, "high": 1.0},
        "extreme_10:3:1": {"low": 10.0, "medium": 3.0, "high": 1.0},
    }

    results = {}

    for config_name, ratios in imbalance_configs.items():
        # Resample to match target ratios
        resampled_texts = []
        resampled_labels = []

        for label, ratio in ratios.items():
            subset = scenarios[scenarios[label_col] == label]
            n_target = max(1, int(len(subset) * ratio))
            sampled = subset.sample(n=n_target, replace=True, random_state=42)
            resampled_texts.extend(sampled[text_col].tolist())
            resampled_labels.extend(sampled[label_col].tolist())

        metrics = _cross_validate(
            resampled_texts, resampled_labels, n_splits=min(n_splits, 3)
        )
        metrics["config"] = config_name
        metrics["n_samples"] = len(resampled_texts)
        metrics["class_distribution"] = {
            label: resampled_labels.count(label) for label in set(resampled_labels)
        }
        results[config_name] = metrics

    return results


# ── Domain shift evaluation ─────────────────────────────────────────────────

def domain_shift_analysis(
    train_scenarios: pd.DataFrame,
    eval_scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    augment_train: bool = True,
) -> Dict:
    """Evaluate domain shift: train on augmented data, test on original.

    This directly measures the gap between internal and external validation,
    quantifying the distribution shift introduced by augmentation.
    """
    train_texts = list(train_scenarios[text_col])
    train_labels = list(train_scenarios[label_col])

    if augment_train:
        for _, row in train_scenarios.iterrows():
            for fn in [synonym_replace, inject_typo, negate]:
                train_texts.append(fn(row[text_col]))
                train_labels.append(row[label_col])

    eval_texts = list(eval_scenarios[text_col])
    eval_labels = list(eval_scenarios[label_col])

    # Train on augmented, test on original
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_eval = vectorizer.transform(eval_texts)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X_train, train_labels)

    train_preds = model.predict(X_train)
    eval_preds = model.predict(X_eval)

    return {
        "train_accuracy": float(accuracy_score(train_labels, train_preds)),
        "eval_accuracy": float(accuracy_score(eval_labels, eval_preds)),
        "performance_gap": float(
            accuracy_score(train_labels, train_preds)
            - accuracy_score(eval_labels, eval_preds)
        ),
        "train_report": classification_report(
            train_labels, train_preds, output_dict=True
        ),
        "eval_report": classification_report(
            eval_labels, eval_preds, output_dict=True
        ),
        "eval_confusion": confusion_matrix(
            eval_labels, eval_preds, labels=["low", "medium", "high"]
        ).tolist(),
    }


# ── Threshold calibration analysis ──────────────────────────────────────────

def threshold_analysis(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """Evaluate effect of classification thresholds on high-risk detection.

    For safety-critical applications, adjusting the decision threshold
    for the high-risk class trades sensitivity for specificity.
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    texts = list(scenarios[text_col])
    labels = list(scenarios[label_col])

    # Augment for training
    aug_texts = list(texts)
    aug_labels = list(labels)
    for _, row in scenarios.iterrows():
        for fn in [synonym_replace, inject_typo, negate]:
            aug_texts.append(fn(row[text_col]))
            aug_labels.append(row[label_col])

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(aug_texts)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    model.fit(X, aug_labels)

    # Get probabilities on original (non-augmented) data
    X_eval = vectorizer.transform(texts)
    probas = model.predict_proba(X_eval)
    classes = list(model.classes_)
    high_idx = classes.index("high") if "high" in classes else -1

    results = {}
    for threshold in thresholds:
        # Binary: high vs not-high
        binary_truth = ["high" if l == "high" else "not-high" for l in labels]
        if high_idx >= 0:
            high_probs = probas[:, high_idx]
            binary_pred = [
                "high" if p >= threshold else "not-high" for p in high_probs
            ]
        else:
            binary_pred = ["not-high"] * len(labels)

        tp = sum(1 for t, p in zip(binary_truth, binary_pred) if t == "high" and p == "high")
        fp = sum(1 for t, p in zip(binary_truth, binary_pred) if t == "not-high" and p == "high")
        fn = sum(1 for t, p in zip(binary_truth, binary_pred) if t == "high" and p == "not-high")
        tn = sum(1 for t, p in zip(binary_truth, binary_pred) if t == "not-high" and p == "not-high")

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        results[f"threshold_{threshold:.1f}"] = {
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "ppv": ppv,
            "npv": npv,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        }

    return results


# ── Cross-validation helper ─────────────────────────────────────────────────

def _cross_validate(
    texts: List[str],
    labels: List[str],
    n_splits: int = 5,
    model=None,
) -> Dict:
    """Perform stratified cross-validation and return aggregate metrics."""
    if model is None:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)

    texts_arr = np.array(texts)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    all_preds = []
    all_truths = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(texts_arr[train_idx])
        X_test = vectorizer.transform(texts_arr[test_idx])
        y_train = labels_arr[train_idx]
        y_test = labels_arr[test_idx]

        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)

        fold_acc = float(accuracy_score(y_test, preds))
        fold_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

        fold_metrics.append({"fold": fold_idx + 1, "accuracy": fold_acc, "f1_macro": fold_f1})
        all_preds.extend(preds)
        all_truths.extend(y_test)

    overall_acc = float(accuracy_score(all_truths, all_preds))
    overall_f1 = float(f1_score(all_truths, all_preds, average="macro", zero_division=0))
    overall_precision = float(precision_score(all_truths, all_preds, average="macro", zero_division=0))
    overall_recall = float(recall_score(all_truths, all_preds, average="macro", zero_division=0))

    return {
        "accuracy": overall_acc,
        "f1_macro": overall_f1,
        "precision_macro": overall_precision,
        "recall_macro": overall_recall,
        "fold_metrics": fold_metrics,
        "confusion_matrix": confusion_matrix(
            all_truths, all_preds, labels=["low", "medium", "high"]
        ).tolist(),
    }


# ── Comprehensive analysis runner ───────────────────────────────────────────

def run_all_sensitivity_analyses(
    scenarios: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> Dict:
    """Run all sensitivity analyses and return consolidated results.

    This is the main entry point for generating the sensitivity analysis
    results reported in the revised manuscript.
    """
    logger.info("Starting comprehensive sensitivity analyses...")

    results = {}

    logger.info("1/5: Augmentation strategy sensitivity")
    results["augmentation_strategies"] = augmentation_sensitivity_analysis(
        scenarios, text_col, label_col
    )

    logger.info("2/5: Augmentation multiplier analysis")
    results["augmentation_multipliers"] = augmentation_multiplier_analysis(
        scenarios, text_col, label_col
    )

    logger.info("3/5: Model comparison")
    results["model_comparison"] = model_comparison_analysis(
        scenarios, text_col, label_col
    )

    logger.info("4/5: Class imbalance robustness")
    results["imbalance_robustness"] = imbalance_robustness_analysis(
        scenarios, text_col, label_col
    )

    logger.info("5/5: Threshold calibration")
    results["threshold_analysis"] = threshold_analysis(
        scenarios, text_col, label_col
    )

    logger.info("All sensitivity analyses complete.")
    return results
