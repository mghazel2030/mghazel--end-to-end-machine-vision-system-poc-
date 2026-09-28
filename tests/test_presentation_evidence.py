"""Unit tests for presentation-oriented evidence helpers."""
from vision_poc.presentation_evidence import _part_metrics


def test_part_metrics_perfect_classifier() -> None:
    """A perfect binary classifier must report unit normalized scores."""
    metrics = _part_metrics(tp=8, tn=2, fp=0, fn=0)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_accept_rate"] == 0.0
    assert metrics["false_reject_rate"] == 0.0
