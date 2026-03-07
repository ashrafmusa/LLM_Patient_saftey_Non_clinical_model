# Contributing to Patient Safety & Quality LLM Project

Thank you for your interest in contributing! This document provides guidelines for participating in this open-source project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We pledge to:

- Be respectful and professional in all interactions
- Focus on collaborative problem-solving
- Welcome diverse perspectives and experiences
- Value constructive criticism and feedback
- Maintain confidentiality of sensitive healthcare data

### Our Standards

Examples of behavior that contributes to creating a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing opinions and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior:
- Harassment, discrimination, or offensive comments
- Publishing others' private information without consent
- Dismissive or disrespectful conduct
- Any conduct that could be harmful to the community

---

## How to Contribute

### Reporting Bugs

Before creating a bug report, please check the issue tracker to avoid duplicates.

**When reporting a bug, include**:
- Clear, descriptive title
- Detailed description of the bug
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (Python version, OS, installed packages)
- Error messages and stack traces (if applicable)

**Example**:
```
Title: Cross-validation fails with single-class data

Environment: Python 3.11, scikit-learn 1.3.0

Steps to reproduce:
1. Generate scenarios with all same label
2. Run train_cv()
3. Observe error

Error:
ValueError: could not convert string to float: 'nan'
```

### Requesting Features

**For feature requests, describe**:
- Use case and problem you're trying to solve
- Proposed solution and alternatives considered
- Potential impact on the project
- Any relevant literature or examples

**Example**:
```
Title: Add support for continuous risk scoring

Use Case: Some clinical applications require fine-grained risk scores 
rather than categorical levels (low/medium/high).

Proposed Solution: Extend risk_assessment.py to output probability scores 
and add optional thresholding for backwards compatibility.
```

### Contributing Code

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** (see Coding Standards below)
4. **Write tests** for new functionality
5. **Update documentation**
6. **Submit a pull request** with clear description

---

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (venv or conda)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/your-username/patient_safety_llm.git
cd patient_safety_llm

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies (including dev tools)
pip install -r requirements.txt
pip install -r requirements-dev.txt
pre-commit install

# 4. Verify setup
make check
```

---

## Submitting Changes

### Pull Request Process

1. **Update your feature branch** with latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run tests locally**:
   ```bash
    make test
   ```

3. **Format your code**:
   ```bash
    make format
    make lint
    make typecheck
    pre-commit run --all-files
   ```

4. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a pull request** with:
   - Clear title (e.g., "Add fairness evaluation module")
   - Detailed description of changes
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist of testing completed

### PR Checklist

```markdown
- [ ] Tests pass locally: `pytest tests/ -v`
- [ ] Code is formatted and linted: `make lint`
- [ ] Type checks pass: `make typecheck`
- [ ] Type hints added where possible
- [ ] Docstrings updated
- [ ] README updated (if needed)
- [ ] No breaking changes (or documented in PR)
```

---

## Coding Standards

### Style Guide

Follow PEP 8 with these guidelines:

```python
# 1. Use descriptive names
good_name = "assess_risk"
bad_name = "ar"

# 2. Add docstrings to all functions/classes
def assess_risk(text: str) -> dict:
    """
    Assess patient safety risk level for clinical text.
    
    Args:
        text: De-identified clinical narrative
        
    Returns:
        Dictionary with keys: risk_level, score, model_based
        
    Example:
        >>> result = assess_risk("Medication error: 10mg ordered, 100mg given")
        >>> result['risk_level']
        'high'
    """

# 3. Use type hints
from typing import Dict, List, Optional

def evaluate(
    n: int = 200,
    output_dir: str = 'reports/eval'
) -> Dict[str, str]:
    """Evaluate risk assessment pipeline."""

# 4. Use meaningful variable names
accuracy = tp / (tp + tn + fp + fn)  # Better than: acc = a/(a+b+c+d)

# 5. Add comments for complex logic
# Use stratified split to maintain class balance across folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### Module Organization

```
src/
├── __init__.py
├── config.py          # Configuration constants
├── deid.py            # De-identification
├── data_ingest.py     # Data loading
├── risk_assessment.py # Core logic
├── explain.py         # Explainability
├── evaluate.py        # Evaluation
└── plots.py           # Visualization
```

### Import Order

```python
# 1. Standard library
import os
import sys
from pathlib import Path

# 2. Third-party packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 3. Local imports
from src.config import DATA_DIR
from src.deid import deidentify_text
```

---

## Testing

### Writing Tests

```python
# tests/test_risk_assessment.py
import pytest
from src.risk_assessment import assess_risk

class TestAssessRisk:
    """Test suite for risk assessment module."""
    
    def test_high_risk_detection(self):
        """Test that high-risk scenarios are detected."""
        text = "Medication error: 10mg ordered, 100mg administered"
        result = assess_risk(text)
        assert result['risk_level'] == 'high'
    
    def test_low_risk_detection(self):
        """Test that low-risk scenarios are detected."""
        text = "Patient denies chest pain. BP normal. No concerns."
        result = assess_risk(text)
        assert result['risk_level'] == 'low'
    
    def test_invalid_input(self):
        """Test handling of invalid input."""
        with pytest.raises(TypeError):
            assess_risk(None)

def test_integration():
    """Test end-to-end pipeline."""
    from src.evaluate import evaluate
    result = evaluate(n=10)  # Small test set
    assert 'accuracy' in result['metrics']
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_risk_assessment.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run only new/changed tests
pytest tests/ -k "test_risk" -v
```

---

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def train_cv(
    input_csv: str,
    n_splits: int = 5,
    augment_multiplier: int = 3
) -> dict:
    """
    Train risk classifier using cross-validation.
    
    Implements 5-fold stratified cross-validation with per-fold TF-IDF 
    vectorization to prevent data leakage. Applies class balancing and 
    probability calibration.
    
    Args:
        input_csv: Path to CSV with 'text' and 'label' columns
        n_splits: Number of CV folds (default: 5)
        augment_multiplier: Data augmentation factor (default: 3)
        
    Returns:
        Dictionary containing:
            - cv_report: Per-fold metrics
            - eval_csv: Path to aggregated evaluation CSV
            - metrics: Overall accuracy, AUC, calibration scores
            - plots: Paths to generated figures
            
    Raises:
        FileNotFoundError: If input_csv does not exist
        ValueError: If CSV missing 'text' or 'label' columns
        
    Example:
        >>> result = train_cv('data/scenarios.csv', n_splits=5)
        >>> print(f"Accuracy: {result['metrics']['accuracy']}")
        Accuracy: 0.95
        
    Notes:
        - Vectorizer is fit per fold to prevent data leakage
        - Random state is fixed (seed=42) for reproducibility
        - Large augment_multiplier may require more memory
    """
```

### README Updates

If adding a feature, update:
1. **README.md** — Add to Features section
2. **Module docstring** — Explain functionality
3. **Tests** — Ensure coverage
4. **Examples** — Show usage

### Changelog Entries

When contributing, add entry to `CHANGELOG.md`:

```markdown
## [Unreleased]

### Added
- Fairness evaluation module for demographic performance analysis
- Continuous risk scoring option (alternative to categorical)

### Changed
- Improved de-identification regex patterns
- Enhanced error messages for debugging

### Fixed
- Cross-validation data leakage bug
- Missing figure path handling
```

---

## Priority Areas for Contribution

### High Priority
- ✅ Real-world data validation (EHR datasets)
- ✅ Fairness and bias analysis
- ✅ Clinician feedback integration
- ✅ Production deployment guide

### Medium Priority
- 📊 Transformer-based models
- 📊 Multi-lingual support
- 📊 Advanced explainability (SHAP, LIME)
- 📊 Performance optimization

### Lower Priority
- 📝 UI enhancements
- 📝 Extended documentation
- 📝 Additional examples/tutorials

---

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Create a GitHub Issue with bug label
- **Features**: Create an Issue with enhancement label
- **Documentation**: Submit a PR with doc changes

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md
- GitHub contributors page
- Project documentation

Thank you for helping make patient safety evaluation more accessible! 🎉
