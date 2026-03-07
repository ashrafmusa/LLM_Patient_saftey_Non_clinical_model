import pytest
from src.evaluate import evaluate


def test_evaluate_smoke(tmp_path):
    # run a small evaluation to smoke-test the pipeline
    out = evaluate(n=10, output_dir=str(tmp_path / "eval"))
    assert 'out_dir' in out
    assert 'results_csv' in out
    assert 'metrics' in out
