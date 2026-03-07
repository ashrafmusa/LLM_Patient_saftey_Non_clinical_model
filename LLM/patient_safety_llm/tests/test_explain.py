import os
import pytest

pytest.importorskip('sklearn')

from src.train import train
from src.explain import explain_text


def test_explain_smoke(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text('id,text\n1,Patient had an error with medication dosing\n2,No adverse events reported\n')
    args = type('A', (), {'input': str(p), 'text_col': 'text', 'label_col': None, 'extra_names_col': None})
    train(args)

    res = explain_text('Medication error occurred')
    assert isinstance(res, dict)
    # explanation may be unavailable if vectorizer/model failed, but function should return a dict
