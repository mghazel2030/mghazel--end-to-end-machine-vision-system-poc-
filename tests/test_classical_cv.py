"""Step #3 classical computer-vision tests."""
from pathlib import Path

from vision_poc.classical_cv import inspect_geometry, localize_part, preprocess_image
from vision_poc.configuration import load_config
from vision_poc.synthetic import generate_sample

CONFIG = Path(__file__).resolve().parents[1] / "config/system.yaml"


def test_localization_recovers_synthetic_part() -> None:
    """Verify a generated part can be localized with plausible geometry."""
    cfg = load_config(CONFIG)
    sample = generate_sample(0, cfg, 1234)
    result = localize_part(preprocess_image(sample.image))
    assert result.size_wh[0] > result.size_wh[1] > 100
    assert abs(result.angle_deg) <= 25


def test_normal_part_geometry_is_rectangular() -> None:
    """Verify normal synthetic parts retain strong rectangularity."""
    cfg = load_config(CONFIG)
    sample = generate_sample(0, cfg, 2222)
    loc = localize_part(preprocess_image(sample.image))
    metrics = inspect_geometry(loc, sample.image.shape)
    assert metrics["rectangularity"] > 0.95
    assert metrics["solidity"] > 0.97
