"""Cumulative Step 1 engineering calculations.
Author: mghazel | Submitted to: Ascension Automation Solutions Ltd. | Version: 2026-09-25
"""
from dataclasses import asdict, dataclass
from math import cos, radians, sin
from typing import Any


@dataclass(frozen=True)
class EngineeringResult:
    """Step 1 metrics and feasibility checks."""
    metrics: dict[str, float]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable representation."""
        return asdict(self)


def positive(value: float, name: str) -> float:
    """Validate a positive scalar."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def rotated_envelope(width: float, height: float, angle_deg: float) -> tuple[float, float]:
    """Return axis-aligned envelope of a rotated rectangle."""
    positive(width, "width")
    positive(height, "height")
    t = radians(abs(angle_deg))
    return width * cos(t) + height * sin(t), width * sin(t) + height * cos(t)


def calculate(cfg: dict[str, Any]) -> EngineeringResult:
    """Calculate sampling, FOV and motion feasibility."""
    a, c, m = cfg["application"], cfg["camera"], cfg["motion"]
    ew, eh = rotated_envelope(a["part_width_mm"], a["part_height_mm"], a["max_rotation_deg"])
    sx = a["fov_width_mm"] / c["width_px"]
    sy = a["fov_height_mm"] / c["height_px"]
    worst = max(sx, sy)
    scratch = a["min_scratch_width_mm"] / worst
    blur = m["speed_mm_s"] * m["exposure_us"] * 1e-6 / worst
    return EngineeringResult(
        {"rotated_width_mm": ew, "rotated_height_mm": eh, "sampling_x_mm_px": sx, "sampling_y_mm_px": sy,
         "scratch_pixels": scratch, "motion_blur_px": blur},
        {"fov": ew + 2 * a["translation_mm"] <= a["fov_width_mm"] and eh + 2 * a["translation_mm"] <= a[
            "fov_height_mm"],
         "scratch_sampling": scratch >= a["min_pixels_scratch"], "motion_blur": blur <= m["max_blur_px"]})
