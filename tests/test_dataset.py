"""Step 2 dataset integration tests."""
import json
from copy import deepcopy
from pathlib import Path

from vision_poc.configuration import load_config
from vision_poc.dataset import generate_dataset

CONFIG = Path(__file__).resolve().parents[1] / "config/system.yaml"


def test_small_dataset_pipeline(tmp_path):
    c = deepcopy(load_config(CONFIG))
    c["dataset"]["total_images"] = 20
    s = generate_dataset(c, tmp_path)
    assert s["total"] == 20
    assert (tmp_path / "03_annotations/annotations.jsonl").exists()
    assert (tmp_path / "04_qa/dataset_montage.png").exists()
    assert len((tmp_path / "03_annotations/annotations.jsonl").read_text().splitlines()) == 20
    assert json.loads((tmp_path / "dataset_summary.json").read_text())["total"] == 20
