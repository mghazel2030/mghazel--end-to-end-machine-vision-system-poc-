"""Logging utilities.
Author: mghazel | Submitted to: Ascension Automation Solutions Ltd. | Version: 2026-09-25
"""
import logging
from pathlib import Path


def configure_logging(log_path:Path)->logging.Logger:
    """Configure console and persistent logging.

    Args:
        log_path: Destination log.
    Returns:
        Project logger.
    Raises:
        OSError: Log destination cannot be created.
    """
    log_path.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
      handlers=[logging.StreamHandler(),logging.FileHandler(log_path,encoding="utf-8")],force=True)
    return logging.getLogger("vision_poc")
