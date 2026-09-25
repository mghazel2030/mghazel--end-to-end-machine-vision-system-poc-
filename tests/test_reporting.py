"""Step 1 output persistence smoke test.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from pathlib import Path

from vision_poc.configuration import load_config
from vision_poc.engineering import calculate
from vision_poc.reporting import save_report


def test_saved_artifacts(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "config/system.yaml")
    save_report(calculate(config), tmp_path)
    expected_files = [
        "01_metrics/metrics.csv",
        "02_feasibility/checks.csv",
        "03_visualizations/scratch_sampling.png",
        "03_visualizations/feasibility.png",
    ]
    for name in expected_files:
        assert (tmp_path / name).stat().st_size > 0
