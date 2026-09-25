"""Deterministic quantitative imaging and conveyor design utilities.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25

All lengths are millimetres unless a name explicitly specifies another unit.
Thin-lens results are screening estimates, not verified vendor optical performance.
"""
from dataclasses import asdict, dataclass
from math import ceil, cos, radians, sin
from typing import Any


@dataclass(frozen=True)
class EngineeringResult:
    """Immutable collection of Step 1 design calculations and checks."""
    metrics: dict[str, float]
    checks: dict[str, bool]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize design results for JSON output.

        Args:
            None.
        Returns:
            JSON-compatible dictionary.
        Raises:
            None under normal use.
        """
        return asdict(self)

def positive(value: float, name: str) -> float:
    """Reject nonpositive numerical engineering inputs.

    Args:
        value: Scalar value to validate.
        name: Human-readable parameter name.
    Returns:
        Validated float.
    Raises:
        ValueError: If value is not strictly positive.
    """
    value = float(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, received {value}")
    return value

def rotated_envelope(width_mm: float, height_mm: float, angle_deg: float) -> tuple[float, float]:
    """Compute axis-aligned bounding box of a rotated rectangle.

    Args:
        width_mm: Unrotated part width.
        height_mm: Unrotated part height.
        angle_deg: Magnitude of in-plane rotation in degrees.
    Returns:
        Rotated width and height, in millimetres.
    Raises:
        ValueError: For nonpositive dimensions.
    """
    w, h = positive(width_mm, "width_mm"), positive(height_mm, "height_mm")
    a = radians(abs(angle_deg))
    return w * abs(cos(a)) + h * abs(sin(a)), w * abs(sin(a)) + h * abs(cos(a))

def calculate(config: dict[str, Any]) -> EngineeringResult:
    """Calculate sampling, motion, lens, bandwidth and timing feasibility.

    Processing workflow:
        1. Validate inputs and worst-case part envelope.
        2. Derive horizontal/vertical object-space sampling.
        3. Compute pixels per defect and required sensor dimensions.
        4. Bound motion blur and conveyor cadence.
        5. Estimate sensor dimensions, focal length and working distance.
        6. Estimate raw bandwidth and exposure-limited illumination needs.
        7. Generate pass/fail feasibility checks and engineering warnings.

    Args:
        config: Validated dictionary loaded from config/system.yaml.
    Returns:
        EngineeringResult with calculated metrics, checks and warnings.
    Raises:
        KeyError: For absent required numerical fields.
        ValueError: For nonpositive required numerical values.
    """
    p, i, s, c, l = (config[k] for k in ("part", "inspection", "station", "camera", "lens"))
    w = positive(p["width_mm"], "part width")
    h = positive(p["height_mm"], "part height")
    fov_w = positive(s["fov_width_mm"], "FOV width")
    fov_h = positive(s["fov_height_mm"], "FOV height")
    nx = int(positive(c["width_px"], "camera width"))
    ny = int(positive(c["height_px"], "camera height"))
    speed = positive(s["conveyor_speed_mm_s"], "conveyor speed")
    defect = positive(i["minimum_scratch_width_mm"], "scratch width")
    min_px = positive(i["min_pixels_across_scratch"], "minimum defect pixels")
    pitch = positive(p["pitch_mm"], "part pitch")
    pixel_um = positive(c["pixel_pitch_um"], "pixel pitch")
    focal = positive(l["focal_length_mm"], "lens focal length")
    bit_depth = positive(s["pixel_depth_bits"], "pixel depth")
    exposure_us = positive(s["nominal_exposure_us"], "exposure")
    budget_ms = positive(i["processing_budget_ms"], "processing budget")
    max_blur = positive(i["max_motion_blur_pixels"], "blur budget")
    translation = float(p["max_translation_mm"])
    if translation < 0:
        raise ValueError("max_translation_mm must be >= 0")
    # The envelope can peak at an interior angle: sample full allowable range, not endpoint only.
    angles = [p["max_rotation_deg"] * k / 1000 for k in range(1001)]
    env_w = max(rotated_envelope(w, h, a)[0] for a in angles)
    env_h = max(rotated_envelope(w, h, a)[1] for a in angles)
    px_x, px_y = fov_w / nx, fov_h / ny
    sensor_w, sensor_h = nx * pixel_um / 1000, ny * pixel_um / 1000
    # Approximate thin-lens working distance: WD ≈ f * (1 + FOV/sensor_width).
    wd_x = focal * (1 + fov_w / sensor_w)
    wd_y = focal * (1 + fov_h / sensor_h)
    focal_x = positive(s["nominal_working_distance_mm"], "working distance") * sensor_w / fov_w
    max_exposure_us = max_blur * min(px_x, px_y) / speed * 1e6
    blur_mm = speed * exposure_us / 1e6
    part_rate_hz = speed / pitch
    # Worst-case raw throughput uses full frames and nominal maximum camera frame rate.
    nominal_fps = positive(c["rated_fps_nominal"], "camera fps")
    raw_mb_frame = nx * ny * bit_depth / 8 / 1e6
    metrics = {
        "worst_rotated_width_mm": env_w,
        "worst_rotated_height_mm": env_h,
        "horizontal_mm_per_pixel": px_x,
        "vertical_mm_per_pixel": px_y,
        "minimum_scratch_pixels_horizontal": defect / px_x,
        "minimum_scratch_pixels_vertical": defect / px_y,
        "minimum_sensor_width_px": ceil(fov_w * min_px / defect),
        "minimum_sensor_height_px": ceil(fov_h * min_px / defect),
        "motion_blur_mm": blur_mm,
        "motion_blur_px_worst_axis": blur_mm / min(px_x, px_y),
        "maximum_exposure_us_for_blur_budget": max_exposure_us,
        "part_arrival_rate_hz": part_rate_hz,
        "time_between_parts_ms": 1000 / part_rate_hz,
        "sensor_width_mm": sensor_w,
        "sensor_height_mm": sensor_h,
        "thin_lens_working_distance_x_mm": wd_x,
        "thin_lens_working_distance_y_mm": wd_y,
        "thin_lens_focal_at_nominal_wd_mm": focal_x,
        "raw_MB_per_frame": raw_mb_frame,
        "raw_MB_per_second_at_part_rate": raw_mb_frame * part_rate_hz,
        "raw_MB_per_second_at_rated_fps": raw_mb_frame * nominal_fps,
        "processing_budget_ms": budget_ms,
        "processing_budget_utilization_at_part_rate": budget_ms * part_rate_hz / 1000,
    }
    checks = {
        "part_envelope_fits_fov_with_translation": env_w + 2 * translation <= fov_w and env_h + 2 * translation <= fov_h,
        "scratch_has_at_least_target_pixels_both_axes": min(defect / px_x, defect / px_y) >= min_px,
        "exposure_meets_motion_blur_budget": exposure_us <= max_exposure_us,
        "camera_frame_rate_exceeds_part_rate": nominal_fps >= part_rate_hz,
        "processing_budget_below_part_interval": budget_ms < 1000 / part_rate_hz,
        "lens_sensor_format_nominally_compatible": str(l["supported_format"]) == str(c["sensor_format"]),
    }
    warnings = [
        "Synthetic image scores are NOT evidence of real-factory defect recall or false-reject rate.",
        "Lens thin-lens estimates exclude distortion, working-distance definitions, tolerance and MTF.",
        "Confirm actual FOV/working distance, MTF, DOF and light level with vendor tools and a bench test.",
        "Exposure requires strobe energy and controller timing validation; ambient light may be insufficient.",
        "Validate camera interface, NIC, packet loss, PLC timing and reject mechanism during integration.",
    ]
    warnings += [f"FAILED REQUIREMENT: {name}" for name, passed in checks.items() if not passed]
    return EngineeringResult(metrics, checks, warnings)
