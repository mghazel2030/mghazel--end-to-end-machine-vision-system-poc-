"""Cumulative Step #3 driver: engineering, synthetic data, OpenCV, and PyTorch.

Processing workflow:
1. Load and validate configuration.
2. Re-run Step #1 quantitative engineering checks.
3. Re-generate Step #2 deterministic synthetic data and exact labels.
4. Demonstrate OpenCV localization/rectification/geometry inspection.
5. Train/validate/test the compact PyTorch U-Net for scratch segmentation.
6. Run hybrid inference and save module-by-module intermediate results.
7. Save quantitative Step #3 reports and logs.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import argparse
import json
from pathlib import Path

import cv2
import torch

from vision_poc.classical_cv import inspect_geometry, localize_part, preprocess_image
from vision_poc.configuration import load_config
from vision_poc.dataset import generate_dataset
from vision_poc.engineering import calculate
from vision_poc.hybrid import inspect_image
from vision_poc.logging_utils import configure_logging
from vision_poc.model import TinyUNet
from vision_poc.reporting import save_report, save_step2_report
from vision_poc.training import train_model

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Execute the complete cumulative Step #3 proof-of-concept.

    Returns:
        Zero for a feasible successful run; one for a handled failure.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/system.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/step_03")
    parser.add_argument("--skip-training", action="store_true", help="Run CV/data pipeline without PyTorch training.")
    args = parser.parse_args()
    logger = configure_logging(ROOT / "logs/step_03.log")
    try:
        cfg = load_config(args.config)
        engineering = calculate(cfg)
        save_report(engineering, args.output / "00_step_01_engineering")
        summary = generate_dataset(cfg, args.output)
        save_step2_report(summary, args.output)
        sample_path = next((args.output / "01_images" / "test").glob("*.png"))
        sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        if sample is None:
            raise OSError("Unable to load generated CV demonstration image")
        localization = localize_part(preprocess_image(sample))
        geometry = inspect_geometry(localization, sample.shape)
        cv_summary = {"sample": str(sample_path), "angle_deg": localization.angle_deg, "geometry": geometry}
        cv_dir = args.output / "05_classical_cv"
        cv_dir.mkdir(parents=True, exist_ok=True)
        (cv_dir / "classical_cv_summary.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
        if not args.skip_training:
            training = train_model(cfg, args.output, args.output / "06_pytorch")
            (args.output / "06_pytorch" / "training_summary.json").write_text(
                json.dumps(training, indent=2), encoding="utf-8"
            )
            model = TinyUNet(int(cfg["training"]["base_channels"]))
            model.load_state_dict(
                torch.load(training["model_path"], map_location="cpu", weights_only=True)
            )
            inspect_image(sample, model, cfg, args.output / "07_hybrid_trace")
        logger.info("Step 3 completed: %d synthetic samples", summary["total"])
        return 0 if all(engineering.checks.values()) else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        logger.exception("Step 3 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
