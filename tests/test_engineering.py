"""Cumulative Step 1 regression tests."""
from pathlib import Path

from vision_poc.configuration import load_config
from vision_poc.engineering import calculate, rotated_envelope

CONFIG = Path(__file__).resolve().parents[1] / "config/system.yaml"


def test_reference_design_passes():
    assert all(calculate(load_config(CONFIG)).checks.values())


def test_rotated_envelope():
    w, h = rotated_envelope(200, 120, 20)
    assert w > 200 and h > 120
