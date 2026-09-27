"""Dataset-level calibration, evaluation, visualization, and failure analysis.

integrated evaluation uses the validation split for threshold calibration and preserves the
test split for final synthetic-domain evaluation. Results quantify software
feasibility only; they are not factory acceptance measurements.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-26
"""
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hybrid import inspect_image
from .model import TinyUNet
from .plc import PLCRejectSimulator


def confusion_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float | int]:
    """Compute part-level binary classification metrics.

    Args:
        tp: Defective parts correctly rejected.
        tn: Normal parts correctly accepted.
        fp: Normal parts incorrectly rejected.
        fn: Defective parts incorrectly accepted.

    Returns:
        Counts plus accuracy, precision, recall, specificity, F1, false-reject,
        and false-accept rates. Undefined ratios are returned as zero.
    """
    eps = 1e-12
    total = tp + tn + fp + fn
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / (precision + recall + eps),
        "false_reject_rate": fp / (fp + tn + eps),
        "false_accept_rate": fn / (fn + tp + eps),
    }


def _load_records(dataset_root: Path, split: str) -> list[dict[str, Any]]:
    """Load JSONL annotation records for one deterministic dataset split."""
    path = dataset_root / "03_annotations" / "annotations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [record for record in records if record["split"] == split]


def _run_records(
    records: list[dict[str, Any]],
    dataset_root: Path,
    model: TinyUNet,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run hybrid inspection for annotation records and retain traceable outputs."""
    rows = []
    for record in records:
        image = cv2.imread(str(dataset_root / record["image"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(dataset_root / record["image"])
        result = inspect_image(image, model, cfg)
        rows.append({"record": record, "result": result})
    return rows


def calibrate_scratch_fraction(
    dataset_root: Path,
    model: TinyUNet,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Calibrate the hybrid scratch-area threshold using validation data only.

    All validation classes participate because the calibrated threshold belongs
    to the final fused part-level decision. Geometry decisions remain fixed;
    the sweep changes only the scratch positive-area threshold. Balanced
    accuracy is optimized first so normal acceptance and defect rejection have
    equal influence; F1 and then the larger threshold break ties.

    Args:
        dataset_root: Generated dataset root.
        model: Trained scratch-segmentation model.
        cfg: Project configuration.

    Returns:
        Selected scratch-fraction threshold and validation sweep table.
    """
    records = _load_records(dataset_root, "validation")
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg["training"]["minimum_scratch_fraction"] = 0.0
    rows = _run_records(records, dataset_root, model, probe_cfg)
    fractions = np.asarray([float(row["result"]["scratch_fraction"]) for row in rows], dtype=float)
    geometry = np.asarray([bool(row["result"]["geometry"]["geometry_damage"]) for row in rows], dtype=bool)
    truth = np.asarray([row["record"]["class"] != "normal" for row in rows], dtype=bool)
    candidates = np.unique(np.concatenate(([0.0], fractions, [1.0])))
    sweep = []
    for threshold in candidates:
        pred = np.logical_or(geometry, fractions >= threshold)
        tp = int(np.logical_and(pred, truth).sum())
        tn = int(np.logical_and(~pred, ~truth).sum())
        fp = int(np.logical_and(pred, ~truth).sum())
        fn = int(np.logical_and(~pred, truth).sum())
        metrics = confusion_metrics(tp, tn, fp, fn)
        balanced_accuracy = 0.5 * (float(metrics["recall"]) + float(metrics["specificity"]))
        sweep.append({"threshold": float(threshold), "balanced_accuracy": balanced_accuracy, **metrics})
    best = max(
        sweep,
        key=lambda row: (float(row["balanced_accuracy"]), float(row["f1"]), float(row["threshold"])),
    )
    return {"selected_threshold": float(best["threshold"]), "validation_metrics": best, "sweep": sweep}


def save_calibration_curve(calibration: dict[str, Any], path: Path) -> None:
    """Save validation threshold-versus-F1 calibration evidence."""
    thresholds = [float(row["threshold"]) for row in calibration["sweep"]]
    balanced = [float(row["balanced_accuracy"]) for row in calibration["sweep"]]
    f1_values = [float(row["f1"]) for row in calibration["sweep"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(thresholds, balanced, marker=".", label="Balanced accuracy")
    ax.plot(thresholds, f1_values, marker=".", label="F1")
    ax.axvline(float(calibration["selected_threshold"]), linestyle="--", label="Selected threshold")
    ax.set_xlabel("Minimum scratch positive-pixel fraction")
    ax.set_ylabel("Validation metric")
    ax.set_title("Validation-Only Hybrid Decision Calibration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix(metrics: dict[str, Any], path: Path) -> None:
    """Save a two-class part-level confusion matrix visualization."""
    matrix = np.asarray([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set_xticks([0, 1], labels=["PASS", "REJECT"])
    ax.set_yticks([0, 1], labels=["Normal", "Defective"])
    ax.set_xlabel("Predicted decision")
    ax.set_ylabel("Ground truth")
    ax.set_title("Part-Level Confusion Matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_latency(latencies: list[float], path: Path) -> None:
    """Save the measured software-inference latency distribution."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(latencies, bins=min(15, max(5, len(latencies) // 3)))
    ax.set_xlabel("Inspection latency (ms)")
    ax.set_ylabel("Parts")
    ax.set_title("Held-Out Test Inspection Latency")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_failure_gallery(failures: list[dict[str, Any]], dataset_root: Path, path: Path) -> None:
    """Save representative false-reject/false-accept images for diagnosis."""
    selected = failures[:12]
    if not selected:
        selected = [{"record": r, "predicted": "CORRECT"} for r in _load_records(dataset_root, "test")[:1]]
    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, selected, strict=False):
        record = item["record"]
        image = cv2.imread(str(dataset_root / record["image"]), cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"GT={record['class']} | Pred={item['predicted']}")
        ax.axis("off")
    fig.suptitle("Representative Test Failures (or one correct sample if none)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def evaluate_test_set(
    dataset_root: Path,
    model: TinyUNet,
    cfg: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Evaluate the calibrated hybrid system on the held-out test split.

    Args:
        dataset_root: integrated evaluation result root.
        model: Trained PyTorch scratch model.
        cfg: Configuration with calibrated decision threshold.
        output: Evaluation output directory.

    Returns:
        Part-level metrics, per-class results, latency statistics, failure count,
        and simulated PLC event count.
    """
    output.mkdir(parents=True, exist_ok=True)
    records = _load_records(dataset_root, "test")
    rows = _run_records(records, dataset_root, model, cfg)
    plc_cfg = cfg["integration"]
    plc = PLCRejectSimulator(float(cfg["motion"]["speed_mm_s"]), float(plc_cfg["reject_distance_mm"]))
    counts = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    latencies = []
    failures = []
    csv_rows = []
    plc_events = []
    for item in rows:
        record, result = item["record"], item["result"]
        defective = record["class"] != "normal"
        rejected = result["decision"] == "REJECT"
        key = "tp" if defective and rejected else "tn" if not defective and not rejected else "fp" if rejected else "fn"
        counts[key] += 1
        class_counts[record["class"]]["total"] += 1
        class_counts[record["class"]]["correct"] += int(defective == rejected)
        latency = float(result["latency_ms"])
        latencies.append(latency)
        event = plc.submit(f"part-{record['index']:05d}", result)
        plc_events.append(event.as_dict())
        csv_rows.append({
            "index": record["index"],
            "class": record["class"],
            "truth": "REJECT" if defective else "PASS",
            "prediction": result["decision"],
            "scratch_fraction": result["scratch_fraction"],
            "geometry_damage": result["geometry"]["geometry_damage"],
            "latency_ms": latency,
            "plc_sequence": event.sequence,
            "reject_delay_ms": event.reject_delay_ms,
        })
        if defective != rejected:
            failures.append({"record": record, "predicted": result["decision"]})
    metrics = confusion_metrics(counts["tp"], counts["tn"], counts["fp"], counts["fn"])
    per_class = {
        name: {"total": values["total"], "correct": values["correct"],
               "correct_rate": values["correct"] / max(values["total"], 1)}
        for name, values in sorted(class_counts.items())
    }
    latency_summary = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies)),
    }
    with (output / "part_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    (output / "plc_events.json").write_text(json.dumps(plc_events, indent=2), encoding="utf-8")
    _save_confusion_matrix(metrics, output / "confusion_matrix.png")
    _save_latency(latencies, output / "latency_distribution.png")
    _save_failure_gallery(failures, dataset_root, output / "failure_gallery.png")
    summary = {
        "test_samples": len(records),
        "metrics": metrics,
        "per_class": per_class,
        "latency": latency_summary,
        "failure_count": len(failures),
        "plc_events": len(plc_events),
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
