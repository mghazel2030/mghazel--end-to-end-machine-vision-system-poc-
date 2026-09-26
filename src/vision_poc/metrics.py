"""Segmentation metrics for Step #3 quantitative validation.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import torch


def binary_segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Compute pixel precision, recall, F1/Dice, and IoU from logits."""
    prediction = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    tp = torch.logical_and(prediction, truth).sum().item()
    fp = torch.logical_and(prediction, ~truth).sum().item()
    fn = torch.logical_and(~prediction, truth).sum().item()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {"precision": precision, "recall": recall, "dice": dice, "iou": iou}
