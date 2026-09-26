"""Cumulative Step #4 driver: train, integrate, calibrate, evaluate, and simulate PLC handoff.

Processing workflow:
1. Load and validate configuration.
2. Re-run Step #1 quantitative engineering checks.
3. Re-generate Step #2 deterministic synthetic data and exact labels.
4. Demonstrate Step #3 OpenCV localization/geometry inspection.
5. Train/validate/test the Step #3 compact PyTorch U-Net.
6. Calibrate the scratch decision threshold on validation data only.
7. Evaluate the complete hybrid inspector on the held-out test split.
8. Simulate ordered PLC/reject events and save traceable outputs.
9. Save quantitative plots, failure cases, reports, and logs.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import torch

from vision_poc.classical_cv import inspect_geometry, localize_part, preprocess_image
from vision_poc.configuration import load_config
from vision_poc.dataset import generate_dataset
from vision_poc.engineering import calculate
from vision_poc.evaluation import (
    calibrate_scratch_fraction,
    evaluate_test_set,
    save_calibration_curve,
)
from vision_poc.hybrid import inspect_image
from vision_poc.logging_utils import configure_logging
from vision_poc.model import TinyUNet
from vision_poc.reporting import save_report, save_step2_report
from vision_poc.training import train_model

ROOT = Path(__file__).resolve().parents[1]


def _load_model(cfg: dict, training: dict) -> TinyUNet:
    """Reconstruct the trained model from the best saved Step #3 checkpoint."""
    model = TinyUNet(int(cfg["training"]["base_channels"]))
    model.load_state_dict(torch.load(training["model_path"], map_location="cpu", weights_only=True))
    model.eval()
    return model


def _load_demo_image(path: Path):
    """Load one grayscale demonstration image or fail with a clear path."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Unable to load generated demonstration image: {path}")
    return image


def main() -> int:
    """Execute the complete cumulative Step #4 proof-of-concept.

    Returns:
        Zero for a feasible successful run; one for a handled failure.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/system.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/step_04")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Run engineering/data/classical-CV only; skip model-dependent Step #4 evaluation.",
    )
    args = parser.parse_args()
    logger = configure_logging(ROOT / "logs/step_04.log")
    try:
        cfg = load_config(args.config)
        engineering = calculate(cfg)
        save_report(engineering, args.output / "00_step_01_engineering")
        dataset_summary = generate_dataset(cfg, args.output)
        save_step2_report(dataset_summary, args.output)

        sample_path = next((args.output / "01_images" / "test").glob("*.png"))
        sample = _load_demo_image(sample_path)
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
            model = _load_model(cfg, training)
            inspect_image(sample, model, cfg, args.output / "07_hybrid_trace")

            calibrated_cfg = copy.deepcopy(cfg)
            calibration = calibrate_scratch_fraction(args.output, model, calibrated_cfg)
            if bool(cfg["integration"]["calibrate_scratch_threshold"]):
                calibrated_cfg["training"]["minimum_scratch_fraction"] = calibration["selected_threshold"]
            integration_dir = args.output / "08_integration_evaluation"
            integration_dir.mkdir(parents=True, exist_ok=True)
            (integration_dir / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            save_calibration_curve(calibration, integration_dir / "calibration_curve.png")
            evaluation = evaluate_test_set(args.output, model, calibrated_cfg, integration_dir)
            latency_target = float(cfg["integration"]["software_latency_target_ms"])
            evaluation["latency_target_ms"] = latency_target
            evaluation["latency_target_met"] = evaluation["latency"]["p95_ms"] <= latency_target
            (integration_dir / "evaluation_summary.json").write_text(
                json.dumps(evaluation, indent=2), encoding="utf-8"
            )
            logger.info(
                "Step 4 held-out evaluation: accuracy=%.3f recall=%.3f FAR=%.3f FRR=%.3f p95=%.1f ms",
                evaluation["metrics"]["accuracy"],
                evaluation["metrics"]["recall"],
                evaluation["metrics"]["false_accept_rate"],
                evaluation["metrics"]["false_reject_rate"],
                evaluation["latency"]["p95_ms"],
            )
        logger.info("Step 4 completed: %d synthetic samples", dataset_summary["total"])
        return 0 if all(engineering.checks.values()) else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration):
        logger.exception("Step 4 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
