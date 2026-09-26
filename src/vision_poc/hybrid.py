"""Hybrid OpenCV + PyTorch inference and traceable result persistence.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .classical_cv import draw_localization, inspect_geometry, localize_part, preprocess_image, rectify_part
from .model import TinyUNet


def inspect_image(
    image: np.ndarray, model: TinyUNet, cfg: dict[str, Any], output: Path | None = None
) -> dict[str, Any]:
    """Run the Step #3 hybrid inspection pipeline on one grayscale image.

    Args:
        image: Input grayscale inspection image.
        model: Trained scratch-segmentation network.
        cfg: Project configuration.
        output: Optional trace directory for intermediate images.

    Returns:
        Localization, geometry, scratch, decision, and latency information.
    """
    start = time.perf_counter()
    preprocessed = preprocess_image(image)
    localization = localize_part(preprocessed)
    geometry = inspect_geometry(localization, image.shape)
    tcfg = cfg["training"]
    network_size = (int(tcfg["input_width_px"]), int(tcfg["input_height_px"]))
    rectified = rectify_part(preprocessed, localization, network_size)
    tensor = torch.from_numpy(rectified.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    device = next(model.parameters()).device
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor.to(device)))[0, 0].cpu().numpy()
    scratch_mask = (probability >= float(tcfg["probability_threshold"])).astype(np.uint8) * 255
    scratch_fraction = float(np.mean(scratch_mask > 0))
    scratch_detected = scratch_fraction >= float(tcfg["minimum_scratch_fraction"])
    reject = bool(geometry["geometry_damage"] or scratch_detected)
    latency_ms = (time.perf_counter() - start) * 1000.0
    result = {
        "center_xy": list(localization.center_xy),
        "angle_deg": localization.angle_deg,
        "geometry": geometry,
        "scratch_fraction": scratch_fraction,
        "scratch_detected": bool(scratch_detected),
        "decision": "REJECT" if reject else "PASS",
        "latency_ms": latency_ms,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output / "01_preprocessed.png"), preprocessed)
        cv2.imwrite(str(output / "02_localization.png"), draw_localization(preprocessed, localization))
        cv2.imwrite(str(output / "03_rectified.png"), rectified)
        cv2.imwrite(str(output / "04_scratch_probability.png"), np.uint8(np.clip(probability * 255, 0, 255)))
        cv2.imwrite(str(output / "05_scratch_mask.png"), scratch_mask)
        (output / "06_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
