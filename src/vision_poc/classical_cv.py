"""Classical computer-vision inspection for localization and geometry.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalizationResult:
    """Localized part pose and binary foreground mask."""

    center_xy: tuple[float, float]
    angle_deg: float
    size_wh: tuple[float, float]
    contour: np.ndarray
    foreground_mask: np.ndarray


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale inspection image for deterministic CV.

    Args:
        image: Single-channel uint8 inspection image.

    Returns:
        Contrast-normalized and lightly denoised uint8 image.

    Raises:
        ValueError: If the image is empty or not single-channel.
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image")
    denoised = cv2.GaussianBlur(image, (5, 5), 0.8)
    return cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)


def localize_part(image: np.ndarray) -> LocalizationResult:
    """Localize the bright rectangular part and estimate its pose.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        Pose, largest contour, and foreground mask.

    Raises:
        ValueError: If no valid part contour can be found.
    """
    threshold = max(70, int(np.percentile(image, 35) + 25))
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Part localization failed")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.10 * image.size:
        raise ValueError("Localized contour is too small to be the part")
    rect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return LocalizationResult((cx, cy), angle, (rw, rh), contour, mask)


def rectify_part(image: np.ndarray, localization: LocalizationResult, output_size: tuple[int, int]) -> np.ndarray:
    """Rotate and crop the localized part into a canonical inspection ROI.

    Args:
        image: Original/preprocessed grayscale image.
        localization: Estimated part pose.
        output_size: Canonical output width and height in pixels.

    Returns:
        Rectified grayscale part image.
    """
    center = localization.center_xy
    matrix = cv2.getRotationMatrix2D(center, localization.angle_deg, 1.0)
    rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=0)
    width = max(8, int(round(localization.size_wh[0])))
    height = max(8, int(round(localization.size_wh[1])))
    crop = cv2.getRectSubPix(rotated, (width, height), center)
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)


def inspect_geometry(localization: LocalizationResult, image_shape: tuple[int, int]) -> dict[str, float | bool]:
    """Estimate edge/corner integrity from contour-to-ideal-rectangle agreement.

    Args:
        localization: Part localization result.
        image_shape: Image height and width.

    Returns:
        Geometry metrics and deterministic damage decision.
    """
    contour_area = float(cv2.contourArea(localization.contour))
    box_area = max(float(localization.size_wh[0] * localization.size_wh[1]), 1.0)
    rectangularity = float(np.clip(contour_area / box_area, 0.0, 1.0))
    hull = cv2.convexHull(localization.contour)
    hull_area = max(float(cv2.contourArea(hull)), 1.0)
    solidity = float(np.clip(contour_area / hull_area, 0.0, 1.0))
    perimeter = float(cv2.arcLength(localization.contour, True))
    approx = cv2.approxPolyDP(localization.contour, 0.015 * perimeter, True)
    corner_count = int(len(approx))
    damaged = rectangularity < 0.975 or solidity < 0.985 or corner_count != 4
    return {
        "rectangularity": rectangularity,
        "solidity": solidity,
        "corner_count": corner_count,
        "geometry_damage": bool(damaged),
        "part_area_fraction": contour_area / float(image_shape[0] * image_shape[1]),
    }


def draw_localization(image: np.ndarray, localization: LocalizationResult) -> np.ndarray:
    """Create a BGR visualization of the localized part and pose axes."""
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    box = cv2.boxPoints((localization.center_xy, localization.size_wh, localization.angle_deg))
    cv2.polylines(vis, [np.int32(box)], True, (0, 255, 0), 2)
    cx, cy = (int(round(v)) for v in localization.center_xy)
    cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
    return vis
