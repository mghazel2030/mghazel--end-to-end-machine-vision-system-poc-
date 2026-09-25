"""Cumulative engineering and dataset reporting.

This module persists the engineering outputs produced by Step #1 and the
synthetic-dataset summary introduced in Step #2.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .engineering import EngineeringResult


def save_report(result: EngineeringResult, output: Path) -> None:
    """Save cumulative Step #1 engineering results and visualizations.

    The function preserves the Step #1 reporting contract so that subsequent
    project steps remain backward-compatible with the engineering baseline.

    Args:
        result: Engineering calculation result containing quantitative metrics
            and feasibility checks.
        output: Root output directory in which the reporting subdirectories
            and artifacts are created.

    Returns:
        None.

    Raises:
        OSError: If output directories or report files cannot be created or
            written.

    Processing Flow:
        1. Create the engineering-report output directories.
        2. Save all quantitative metrics to CSV.
        3. Save all feasibility checks to CSV.
        4. Save a complete JSON representation of the engineering result.
        5. Generate the scratch-sampling visualization.
        6. Generate the feasibility-check visualization.
    """
    metrics_dir = output / "01_metrics"
    feasibility_dir = output / "02_feasibility"
    visualization_dir = output / "03_visualizations"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    feasibility_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Quantitative engineering metrics
    # ---------------------------------------------------------
    metrics_path = metrics_dir / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for name, value in result.metrics.items():
            writer.writerow([name, value])

    # ---------------------------------------------------------
    # 2. Engineering feasibility checks
    # ---------------------------------------------------------
    checks_path = feasibility_dir / "checks.csv"

    with checks_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["check", "passed"])

        for name, passed in result.checks.items():
            writer.writerow([name, passed])

    # ---------------------------------------------------------
    # 3. Complete machine-readable engineering result
    # ---------------------------------------------------------
    json_path = output / "engineering_results.json"

    json_path.write_text(
        json.dumps(result.as_dict(), indent=2),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 4. Scratch-sampling visualization
    # ---------------------------------------------------------
    scratch_pixels = result.metrics.get("scratch_pixels", 0.0)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        ["Smallest Scratch"],
        [scratch_pixels],
    )

    ax.axhline(
        y=5.0,
        linestyle="--",
        label="Minimum target = 5 pixels",
    )

    ax.set_ylabel("Pixels Across Defect")
    ax.set_title("Smallest-Defect Sampling Verification")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        visualization_dir / "scratch_sampling.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 5. Feasibility-check visualization
    # ---------------------------------------------------------
    check_names = list(result.checks.keys())

    check_values = [
        1 if result.checks[name] else 0
        for name in check_names
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        check_names,
        check_values,
    )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Pass = 1 / Fail = 0")
    ax.set_title("Step #1 Engineering Feasibility Checks")
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()

    fig.savefig(
        visualization_dir / "feasibility.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_step2_report(
    summary: dict[str, Any],
    output: Path,
) -> None:
    """Save a human-readable Step #2 synthetic-dataset report.

    Args:
        summary: Dataset-generation summary containing the total sample count,
            split/class counts, and generated artifact locations.
        output: Root Step #2 output directory.

    Returns:
        None.

    Raises:
        OSError: If the report cannot be written.

    Processing Flow:
        1. Create the report header.
        2. Record the total number of generated images.
        3. Document the available ground-truth annotations.
        4. Record split/class sample counts.
        5. State the synthetic-data validation limitation.
        6. Persist the report as Markdown.
    """
    output.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Step #2 Synthetic Dataset Report",
        "",
        f"- Total images: **{summary['total']}**",
        "- Ground truth: pixel masks, bounding boxes, and class labels.",
        "- Synthetic data does not establish real-factory performance.",
        "",
        "## Split / Class Counts",
        "",
    ]

    lines.extend(
        f"- {name}: {count}"
        for name, count in summary["counts"].items()
    )

    report_path = output / "dataset_report.md"

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )