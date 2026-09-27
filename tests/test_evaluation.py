"""Tests for integrated evaluation part-level evaluation metrics."""
from vision_poc.evaluation import confusion_metrics


def test_confusion_metrics_known_case():
    """Verify binary decision metrics against a hand-computed case."""
    metrics = confusion_metrics(tp=8, tn=9, fp=1, fn=2)
    assert metrics["accuracy"] == 0.85
    assert abs(metrics["recall"] - 0.8) < 1e-9
    assert abs(metrics["specificity"] - 0.9) < 1e-9
    assert abs(metrics["false_reject_rate"] - 0.1) < 1e-9
    assert abs(metrics["false_accept_rate"] - 0.2) < 1e-9
