"""Validated YAML configuration loading.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing or structurally invalid."""

def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the high-level engineering configuration.

    Args:
        path: Path to the Step 1 YAML configuration.

    Returns:
        Dictionary of required configuration sections and their contents.

    Raises:
        ConfigurationError: If file is absent, malformed, or missing required sections.
        OSError: If reading fails for reasons other than a missing file.
    """
    if not path.is_file():
        raise ConfigurationError(f"Configuration not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML: {exc}") from exc
    required = ("part", "inspection", "station", "camera", "lens", "outputs")
    if not isinstance(data, dict) or any(not isinstance(data.get(k), dict) for k in required):
        raise ConfigurationError(f"Expected configuration sections: {required}")
    return data
