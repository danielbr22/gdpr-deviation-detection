import pytest
from src.evaluation.evaluate import _parse_article, aggregate


def test_parse_article():
    assert _parse_article("Art. 20") == 20
    assert _parse_article("Art. 5") == 5
    assert _parse_article("Art. 12") == 12


def test_aggregate_empty():
    result = aggregate([])
    assert result["overall"]["tp"] == 0
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0


def test_aggregate_sums():
    fake_results = [
        {
            "overall": {"tp": 2, "fp": 5, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "type_metrics": {
                "constraint_coverage": {"tp": 1, "fp": 3, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "execution_style": {"tp": 1, "fp": 2, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "negation": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "responsibility": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "data": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
        },
        {
            "overall": {"tp": 1, "fp": 2, "fn": 2, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "type_metrics": {
                "constraint_coverage": {"tp": 1, "fp": 1, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "execution_style": {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "negation": {"tp": 0, "fp": 0, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "responsibility": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "data": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            },
        },
    ]
    agg = aggregate(fake_results)
    assert agg["overall"]["tp"] == 3
    assert agg["overall"]["fp"] == 7
    assert agg["overall"]["fn"] == 3
    assert agg["type_metrics"]["constraint_coverage"]["tp"] == 2
    assert agg["type_metrics"]["execution_style"]["fn"] == 2
