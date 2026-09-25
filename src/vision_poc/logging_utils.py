"""File and console logging with no duplicated handlers.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import logging
from pathlib import Path


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure the project's structured-in-time text logging.

    Args:
        log_path: Destination UTF-8 log file path.
    Returns:
        Configured project logger.
    Raises:
        OSError: If the log directory or file cannot be created.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vision_poc")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
