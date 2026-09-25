"""Human-readable engineering reports and high-level engineering charts.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engineering import EngineeringResult


def save_report(result: EngineeringResult, output: Path) -> None:
    """Save all engineering results and feasibility charts to named subfolders.

    Args:
        result: Deterministic calculations from the engineering module.
        output: Root directory for Step 1 generated outputs.
    Returns:
        None; writes JSON, Markdown, CSV and PNG artifacts.
    Raises:
        OSError: If an output file or directory cannot be written.
    """
    calculations = output / "01_calculations"
    feasibility = output / "02_feasibility"
    figures = output / "03_visualizations"
    for folder in (calculations, feasibility, figures):
        folder.mkdir(parents=True, exist_ok=True)
    (calculations / "engineering_results.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    with (calculations / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(result.metrics.items())
    with (feasibility / "checks.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["requirement", "passed"])
        writer.writerows(result.checks.items())
    md = ["# Step 1 Engineering Report", "", "## Feasibility"]
    md += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in result.checks.items()]
    md += ["", "## Metrics"] + [f"- {k}: {v:.5g}" for k, v in result.metrics.items()]
    md += ["", "## Engineering limitations"] + [f"- {w}" for w in result.warnings]
    (output / "engineering_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    labels = ["X", "Y"]
    values = [result.metrics["minimum_scratch_pixels_horizontal"], result.metrics["minimum_scratch_pixels_vertical"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.axhline(5, linestyle="--", label="5 pixels target")
    ax.set_ylabel("Pixels across 0.5 mm scratch")
    ax.set_title("Defect Sampling by Image Axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "scratch_sampling.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [k.replace("_", " ") for k in result.checks]
    ax.barh(names, [1 if v else 0 for v in result.checks.values()])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Feasibility (1 = pass)")
    ax.set_title("Step 1 Engineering Gates")
    fig.tight_layout()
    fig.savefig(figures / "feasibility.png", dpi=160)
    plt.close(fig)
