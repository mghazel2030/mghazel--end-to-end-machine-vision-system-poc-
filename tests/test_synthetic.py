"""Step 2 generator tests."""
from pathlib import Path

import cv2

from vision_poc.configuration import load_config
from vision_poc.synthetic import generate_sample, save_sample

CONFIG = Path(__file__).resolve().parents[1] / "config/system.yaml"


def test_generation_is_reproducible():
    c = load_config(CONFIG)
    a = generate_sample(1, c, 123)
    b = generate_sample(1, c, 123)
    assert (a.image == b.image).all() and (a.mask == b.mask).all()


def test_scratch_has_mask():
    c = load_config(CONFIG)
    s = generate_sample(1, c, 123)  # index 1 => scratch
    assert s.class_name == "scratch" and s.mask.max() == 255 and s.bbox_xyxy is not None


def test_normal_has_no_mask():
    c = load_config(CONFIG)
    s = generate_sample(0, c, 123)
    assert s.class_name == "normal" and s.mask.max() == 0 and s.bbox_xyxy is None


def test_save_sample(tmp_path):
    c = load_config(CONFIG)
    s = generate_sample(2, c, 123)
    ip = tmp_path / "i.png"
    mp = tmp_path / "m.png"
    save_sample(s, ip, mp)
    assert cv2.imread(str(ip), 0) is not None and cv2.imread(str(mp), 0) is not None
