"""Presentation-oriented evidence generation for the machine-vision PoC.

This module converts intermediate algorithm states and quantitative evaluation
results into reviewable images, CSV tables, JSON summaries, and plots. The
artifacts are intended for engineering review and presentation preparation;
they do not replace factory acceptance testing on representative real data.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-27
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .classical_cv import draw_localization, inspect_geometry, localize_part, preprocess_image, rectify_part
from .hybrid import inspect_image
from .metrics import binary_segmentation_metrics
from .model import TinyUNet
from .torch_data import ScratchSegmentationDataset


def _records(root: Path, split: str | None = None) -> list[dict[str, Any]]:
    """Load annotation records, optionally filtering by dataset split.

    Args:
        root: Runtime result root containing ``03_annotations``.
        split: Optional ``train``, ``validation``, or ``test`` selector.

    Returns:
        Parsed annotation dictionaries in manifest order.

    Raises:
        FileNotFoundError: If the annotation file is unavailable.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-27
    """
    path = root / "03_annotations" / "annotations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return rows if split is None else [row for row in rows if row["split"] == split]


def _load_gray(path: Path) -> np.ndarray:
    """Load one grayscale image and fail explicitly when it cannot be read."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Unable to load image: {path}")
    return image


def save_processing_sequence(root: Path, output: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Save preprocessing/localization/orientation/rectification on one common test image.

    Args:
        root: Runtime result root containing the generated dataset.
        output: Destination directory for presentation evidence.
        cfg: Validated configuration containing canonical network dimensions.

    Returns:
        Metadata describing the selected common image and estimated pose.

    Raises:
        ValueError: If the test split is empty.
        OSError: If an image cannot be loaded or saved.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-27
    """
    output.mkdir(parents=True, exist_ok=True)
    test = _records(root, "test")
    if not test:
        raise ValueError("No test image is available for processing-sequence evidence")
    record = next((row for row in test if row["class"] == "mixed"), test[0])
    original = _load_gray(root / record["image"])
    preprocessed = preprocess_image(original)
    localization = localize_part(preprocessed)
    localized = draw_localization(preprocessed, localization)
    oriented = localized.copy()
    center = tuple(int(round(value)) for value in localization.center_xy)
    length = 80
    angle = np.deg2rad(localization.angle_deg)
    endpoint = (int(center[0] + length * np.cos(angle)), int(center[1] + length * np.sin(angle)))
    cv2.arrowedLine(oriented, center, endpoint, (255, 0, 0), 3, tipLength=0.2)
    cv2.putText(oriented, f"angle={localization.angle_deg:.2f} deg", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    size = (int(cfg["training"]["input_width_px"]), int(cfg["training"]["input_height_px"]))
    rectified = rectify_part(preprocessed, localization, size)
    images = {
        "01_common_test_image.png": original,
        "02_preprocessed.png": preprocessed,
        "03_localization.png": localized,
        "04_orientation.png": oriented,
        "05_rectified.png": rectified,
    }
    for name, image in images.items():
        if not cv2.imwrite(str(output / name), image):
            raise OSError(f"Unable to save presentation image: {output / name}")
    summary = {
        "index": record["index"], "class": record["class"], "image": record["image"],
        "center_xy": localization.center_xy, "angle_deg": localization.angle_deg,
        "size_wh": localization.size_wh,
    }
    (output / "processing_sequence.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _ideal_rectangle_mask(localization, shape: tuple[int, int]) -> np.ndarray:
    """Create the filled ideal rotated rectangle implied by the localized part pose."""
    mask = np.zeros(shape, dtype=np.uint8)
    box = cv2.boxPoints((localization.center_xy, localization.size_wh, localization.angle_deg))
    cv2.fillConvexPoly(mask, np.int32(box), 255)
    return mask


def _geometry_discrepancy(localization) -> np.ndarray:
    """Estimate missing-material pixels as ideal-rectangle support absent from the part mask."""
    ideal = _ideal_rectangle_mask(localization, localization.foreground_mask.shape)
    return cv2.bitwise_and(ideal, cv2.bitwise_not(localization.foreground_mask))


def _part_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    """Calculate binary part-level inspection metrics from confusion counts."""
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / total,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_accept_rate": fn / max(tp + fn, 1),
        "false_reject_rate": fp / max(tn + fp, 1),
    }


def _save_metric_bars(metrics: dict[str, float | int], path: Path, title: str) -> None:
    """Save a bar chart for normalized classification metrics."""
    names = ["accuracy", "precision", "recall", "specificity", "f1"]
    values = [float(metrics[name]) for name in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(names, values)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_classical_geometry(root: Path, output: Path) -> dict[str, Any]:
    """Evaluate classical geometry detection on every held-out test image.

    Part-level detection is evaluated across the complete test split. Pixel-level
    discrepancy metrics are additionally calculated only where unambiguous
    geometry ground truth exists: normal, edge-damage, and corner-damage samples.
    Scratch samples are negative for geometry; mixed samples are excluded from
    pixel metrics because their single synthetic mask combines scratch and geometry.

    Args:
        root: Runtime result root.
        output: Destination for overlays, tables, summaries, and plots.

    Returns:
        Part-level and geometry-pixel-level evaluation summary.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-27
    """
    overlays = output / "overlays_all_test_images"
    overlays.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    pixel_tp = pixel_fp = pixel_fn = 0
    rows = []
    pixel_samples = 0
    for record in _records(root, "test"):
        image = _load_gray(root / record["image"])
        gt = _load_gray(root / record["mask"])
        pre = preprocess_image(image)
        loc = localize_part(pre)
        geometry = inspect_geometry(loc, image.shape)
        pred = bool(geometry["geometry_damage"])
        truth = record["class"] in {"edge_damage", "corner_damage", "mixed"}
        key = "tp" if truth and pred else "tn" if not truth and not pred else "fp" if pred else "fn"
        counts[key] += 1
        discrepancy = _geometry_discrepancy(loc)
        if record["class"] in {"normal", "edge_damage", "corner_damage"}:
            target = gt > 0 if record["class"] != "normal" else np.zeros_like(gt, dtype=bool)
            prediction = discrepancy > 0
            pixel_tp += int(np.logical_and(prediction, target).sum())
            pixel_fp += int(np.logical_and(prediction, ~target).sum())
            pixel_fn += int(np.logical_and(~prediction, target).sum())
            pixel_samples += 1
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vis[gt > 0] = (0, 255, 255)
        vis[discrepancy > 0] = (0, 0, 255)
        cv2.putText(vis, f"GT={record['class']}  CCV={'REJECT' if pred else 'PASS'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlays / f"part_{record['index']:05d}_{record['class']}.png"), vis)
        rows.append({"index": record["index"], "class": record["class"], "truth_geometry": truth,
                     "predicted_geometry": pred, **geometry})
    metrics = _part_metrics(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    pixel_precision = pixel_tp / max(pixel_tp + pixel_fp, 1)
    pixel_recall = pixel_tp / max(pixel_tp + pixel_fn, 1)
    pixel = {
        "evaluated_samples": pixel_samples, "tp_pixels": pixel_tp, "fp_pixels": pixel_fp,
        "fn_pixels": pixel_fn, "precision": pixel_precision, "recall": pixel_recall,
        "dice": 2 * pixel_tp / max(2 * pixel_tp + pixel_fp + pixel_fn, 1),
        "iou": pixel_tp / max(pixel_tp + pixel_fp + pixel_fn, 1),
        "scope": "normal + edge_damage + corner_damage; mixed excluded from pixel metrics",
    }
    with (output / "ccv_test_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _save_metric_bars(metrics, output / "ccv_part_level_metrics.png", "CCV Geometry: Held-Out Part-Level Metrics")
    summary = {
        "test_samples": len(rows),
        "part_level_detection": metrics,
        "pixel_discrepancy": pixel,
        "interpretation": (
            "CCV performs part-level geometry defect detection; the discrepancy mask "
            "is an approximate localization/segmentation aid."
        ),
    }
    (output / "ccv_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_dataset_split_evidence(root: Path, output: Path) -> dict[str, Any]:
    """Save train/validation/test counts and one representative image from each split."""
    output.mkdir(parents=True, exist_ok=True)
    records = _records(root)
    counts = Counter(row["split"] for row in records)
    ai_counts = Counter(row["split"] for row in records if row["class"] in {"normal", "scratch"})
    summary = {split: {"all_images": counts[split], "unet_eligible": ai_counts[split]}
               for split in ("train", "validation", "test")}
    for split in summary:
        candidates = [row for row in records if row["split"] == split and row["class"] == "scratch"]
        record = candidates[0] if candidates else next(row for row in records if row["split"] == split)
        image = _load_gray(root / record["image"])
        cv2.imwrite(str(output / f"sample_{split}_{record['class']}.png"), image)
    with (output / "dataset_split_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "all_images", "unet_eligible_normal_plus_scratch"])
        for split, values in summary.items():
            writer.writerow([split, values["all_images"], values["unet_eligible"]])
    (output / "dataset_split_counts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_unet_architecture_diagram(output: Path) -> None:
    """Render a presentation-ready high-level Tiny U-Net architecture diagram."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axis("off")
    blocks = [(0.05, "Input\n1×H×W"), (0.20, "Encoder 1\nConv + ReLU"), (0.35, "Encoder 2\nDownsample"),
              (0.50, "Bottleneck\nDeep features"), (0.65, "Decoder 2\nUpsample + skip"),
              (0.80, "Decoder 1\nUpsample + skip"), (0.94, "Output\n1×H×W logits")]
    for x, label in blocks:
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=10,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black"})
    for (x1, _), (x2, _) in zip(blocks[:-1], blocks[1:], strict=True):
        ax.annotate("", xy=(x2 - 0.055, 0.55), xytext=(x1 + 0.055, 0.55), arrowprops={"arrowstyle": "->"})
    ax.annotate("skip", xy=(0.76, 0.72), xytext=(0.20, 0.72), arrowprops={"arrowstyle": "->"}, ha="center")
    ax.annotate("skip", xy=(0.62, 0.35), xytext=(0.35, 0.35), arrowprops={"arrowstyle": "->"}, ha="center")
    ax.set_title("Compact U-Net for Scratch Segmentation", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_learning_curves(training: dict[str, Any], output: Path) -> None:
    """Save loss and Dice learning curves as functions of epoch."""
    output.mkdir(parents=True, exist_ok=True)
    history = training["history"]
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train loss")
    ax.plot(epochs, [row["validation_loss"] for row in history], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCEWithLogits loss")
    ax.set_title("U-Net Learning Curve: Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curve_loss.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, [row["train"]["dice"] for row in history], marker="o", label="Train Dice")
    ax.plot(epochs, [row["validation"]["dice"] for row in history], marker="o", label="Validation Dice")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice score")
    ax.set_ylim(0, 1.05)
    ax.set_title("U-Net Learning Curve: Dice")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curve_dice.png", dpi=170)
    plt.close(fig)


def evaluate_unet(root: Path, model: TinyUNet, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Evaluate U-Net segmentation on every eligible held-out normal/scratch test image.

    Pixel confusion counts are accumulated globally before Precision, Recall,
    Dice/F1, IoU, and Accuracy are calculated. This avoids assigning a zero Dice
    score to a correctly predicted all-background normal image.

    Args:
        root: Runtime result root.
        model: Trained scratch segmentation model in evaluation mode.
        cfg: Validated configuration.
        output: Destination for overlays and quantitative results.

    Returns:
        Aggregate pixel-level segmentation metrics and evaluated image count.

    Author: mghazel
    Submitted to: Ascension Automation Solutions Ltd.
    Version: 2026-09-27
    """
    overlays = output / "overlays_all_unet_test_images"
    overlays.mkdir(parents=True, exist_ok=True)
    size = (int(cfg["training"]["input_width_px"]), int(cfg["training"]["input_height_px"]))
    dataset = ScratchSegmentationDataset(root, "test", size)
    device = next(model.parameters()).device
    tp = tn = fp = fn = 0
    rows = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image_t, mask_t, label = dataset[idx]
            logits = model(image_t.unsqueeze(0).to(device))
            per_image = binary_segmentation_metrics(logits.cpu(), mask_t.unsqueeze(0))
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            prediction = probability >= float(cfg["training"]["probability_threshold"])
            image = (image_t[0].numpy() * 255).astype(np.uint8)
            target = mask_t[0].numpy() > 0
            tp += int(np.logical_and(prediction, target).sum())
            tn += int(np.logical_and(~prediction, ~target).sum())
            fp += int(np.logical_and(prediction, ~target).sum())
            fn += int(np.logical_and(~prediction, target).sum())
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            vis[target] = (0, 255, 255)
            vis[prediction] = (0, 0, 255)
            cv2.imwrite(str(overlays / f"unet_{idx:04d}_{label}.png"), vis)
            rows.append({"sample": idx, "class": label, **per_image})
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    aggregate = {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "tp_pixels": tp, "tn_pixels": tn, "fp_pixels": fp, "fn_pixels": fn,
    }
    with (output / "unet_test_metrics_per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _save_metric_bars(
        {"accuracy": aggregate["accuracy"], "precision": precision, "recall": recall,
         "specificity": tn / max(tn + fp, 1), "f1": aggregate["dice"]},
        output / "unet_segmentation_metrics.png", "U-Net: Held-Out Pixel Segmentation Metrics",
    )
    summary = {
        "evaluated_test_images": len(dataset), "scope": "all held-out normal + scratch images",
        "metrics": aggregate,
        "note": "Global pixel confusion counts are used; mixed labels remain excluded because they are not class-pure.",
    }
    (output / "unet_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_fusion_evidence(root: Path, model: TinyUNet, cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Save decision-fusion overlays and metrics for every held-out test image."""
    overlays = output / "overlays_all_test_images"
    overlays.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    rows = []
    for record in _records(root, "test"):
        image = _load_gray(root / record["image"])
        gt = _load_gray(root / record["mask"])
        result = inspect_image(image, model, cfg)
        truth = record["class"] != "normal"
        rejected = result["decision"] == "REJECT"
        key = "tp" if truth and rejected else "tn" if not truth and not rejected else "fp" if rejected else "fn"
        counts[key] += 1
        pre = preprocess_image(image)
        loc = localize_part(pre)
        discrepancy = _geometry_discrepancy(loc)
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        vis[gt > 0] = (0, 255, 255)
        vis[discrepancy > 0] = (0, 0, 255)
        cv2.putText(vis, f"GT={record['class']}  FUSION={result['decision']}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlays / f"fusion_{record['index']:05d}_{record['class']}.png"), vis)
        rows.append({"index": record["index"], "class": record["class"], "decision": result["decision"],
                     "scratch_fraction": result["scratch_fraction"],
                     "geometry_damage": result["geometry"]["geometry_damage"], "latency_ms": result["latency_ms"]})
    metrics = _part_metrics(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    with (output / "fusion_test_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _save_metric_bars(metrics, output / "fusion_part_level_metrics.png", "Hybrid CCV + U-Net: Held-Out Metrics")
    summary = {"test_samples": len(rows), "scope": "all held-out test images", "metrics": metrics}
    (output / "fusion_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
