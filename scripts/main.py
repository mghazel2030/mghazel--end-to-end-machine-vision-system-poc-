"""Cumulative Step 2 driver.

Purpose:
    Run Step 1 engineering validation, then Step 2 deterministic synthetic
    dataset generation, annotations, QA and reporting.

Workflow:
    configuration -> engineering -> dataset generator -> masks/annotations ->
    QA montage/manifest -> reports/logging.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import argparse
from pathlib import Path

from vision_poc.configuration import load_config
from vision_poc.dataset import generate_dataset
from vision_poc.engineering import calculate
from vision_poc.logging_utils import configure_logging
from vision_poc.reporting import save_report, save_step2_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Execute cumulative Step 1 + Step 2.

    Returns:
        0 on successful feasible run; 1 for handled configuration/I/O failure.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "config/system.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "results/step_02")
    args = p.parse_args()
    logger = configure_logging(ROOT / "logs/step_02.log")
    try:
        cfg = load_config(args.config)
        eng = calculate(cfg)
        save_report(eng, args.output / "00_step_01_engineering")
        summary = generate_dataset(cfg, args.output)
        save_step2_report(summary, args.output)
        logger.info("Generated %d samples in %s", summary["total"], args.output)
        return 0 if all(eng.checks.values()) else 1
    except (OSError, ValueError, KeyError, TypeError):
        logger.exception("Step 2 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
