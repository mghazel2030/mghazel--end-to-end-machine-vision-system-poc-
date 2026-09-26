"""Configuration loading/validation.
Author: mghazel | Submitted to: Ascension Automation Solutions Ltd. | Version: 2026-09-25
"""
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised for invalid project configuration."""

def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration.

    Args:
        path: YAML file.
    Returns:
        Validated configuration mapping.
    Raises:
        FileNotFoundError: File absent.
        ConfigurationError: Required content invalid.
        yaml.YAMLError: YAML invalid.
    """
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict):
        raise ConfigurationError("Configuration root must be a mapping")
    for key in ("application", "camera", "motion", "dataset", "training"):
        if key not in data:
            raise ConfigurationError(f"Missing section: {key}")
    d=data["dataset"]
    if d["image_width_px"]<=0 or d["image_height_px"]<=0 or d["total_images"]<5:
        raise ConfigurationError("Invalid dataset dimensions/count")
    return data
