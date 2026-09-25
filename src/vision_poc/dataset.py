"""Dataset generation, splitting, annotations and QA.
Author: mghazel | Submitted to: Ascension Automation Solutions Ltd. | Version: 2026-09-25
"""
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .synthetic import generate_sample, save_sample


def assign_split(index: int, total: int, train: float, val: float) -> str:
    """Assign deterministic train/validation/test partition."""
    r = index / total
    return "train" if r < train else ("validation" if r < train + val else "test")


def generate_dataset(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    """Generate images, masks, JSONL annotations, manifest and QA montage.

    Args:
        cfg: Validated configuration.
        output: Step 2 output root.
    Returns:
        Dataset summary.
    Raises:
        OSError: Output cannot be written.
        ValueError: Split fractions invalid.
    """
    d = cfg["dataset"]
    total = int(d["total_images"])
    train = float(d["train_fraction"])
    val = float(d["validation_fraction"])
    if train <= 0 or val <= 0 or train + val >= 1:
        raise ValueError("Invalid split fractions")
    ann = output / "03_annotations" / "annotations.jsonl"
    ann.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    preview = []
    counts = Counter()
    with ann.open("w", encoding="utf-8") as af:
        for i in range(total):
            split = assign_split(i, total, train, val)
            s = generate_sample(i, cfg, int(d["seed"]) + i)
            ip = output / "01_images" / split / f"part_{i:05d}.png"
            mp = output / "02_masks" / split / f"part_{i:05d}.png"
            save_sample(s, ip, mp)
            counts[(split, s.class_name)] += 1
            rec = {**s.metadata, "split": split, "class": s.class_name, "image": str(ip.relative_to(output)),
                   "mask": str(mp.relative_to(output)), "bbox_xyxy": s.bbox_xyxy}
            af.write(json.dumps(rec) + "\n")
            rows.append(rec)
            if len(preview) < 12:
                preview.append((s.image, s.mask, s.class_name))
    qa = output / "04_qa"
    qa.mkdir(parents=True, exist_ok=True)
    manifest = qa / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        fields = ["index", "seed", "split", "class", "image", "mask", "bbox_xyxy", "rotation_deg", "translation_px"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, (im, mask, label) in zip(axes.flat, preview, strict=False):
        rgb = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        rgb[mask > 0] = [255, 0, 0]
        ax.imshow(rgb)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    montage = qa / "dataset_montage.png"
    fig.savefig(montage, dpi=130)
    plt.close(fig)
    summary = {"total": total, "counts": {"|".join(k): v for k, v in sorted(counts.items())},
               "annotations": str(ann), "manifest": str(manifest), "qa_montage": str(montage)}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
