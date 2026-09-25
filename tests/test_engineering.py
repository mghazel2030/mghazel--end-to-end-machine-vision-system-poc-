"""Step 1 deterministic unit and configuration tests.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from pathlib import Path

import pytest

from vision_poc.configuration import ConfigurationError, load_config
from vision_poc.engineering import calculate, positive, rotated_envelope

CONFIG = Path(__file__).resolve().parents[1] / "config/system.yaml"

def test_reference_configuration_passes():
    result = calculate(load_config(CONFIG))
    assert all(result.checks.values()), result.checks
    assert result.metrics["minimum_scratch_pixels_horizontal"] >= 5

def test_rotated_envelope_increases_width():
    w, h = rotated_envelope(200, 120, 20)
    assert 200 < w < 240
    assert 120 < h < 200

def test_negative_dimension_rejected():
    with pytest.raises(ValueError):
        positive(-1, "test")

def test_missing_configuration_rejected(tmp_path):
    with pytest.raises(ConfigurationError):
        load_config(tmp_path / "missing.yaml")

def test_inadequate_camera_fails():
    config = load_config(CONFIG)
    config["camera"]["width_px"] = 1000
    result = calculate(config)
    assert not result.checks["scratch_has_at_least_target_pixels_both_axes"]

def test_excessive_exposure_fails():
    config = load_config(CONFIG)
    config["station"]["nominal_exposure_us"] = 1000
    result = calculate(config)
    assert not result.checks["exposure_meets_motion_blur_budget"]

def test_part_misalignment_fails():
    config = load_config(CONFIG)
    config["part"]["max_translation_mm"] = 20
    result = calculate(config)
    assert not result.checks["part_envelope_fits_fov_with_translation"]
