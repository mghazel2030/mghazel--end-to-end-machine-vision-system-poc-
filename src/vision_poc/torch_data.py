"""PyTorch dataset utilities for synthetic scratch segmentation.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ScratchSegmentationDataset(Dataset):
    """Load synthetic images and scratch-relevant masks from JSONL annotations."""

    def __init__(self, results_root: Path, split: str, image_size: tuple[int, int] = (256, 208)) -> None:
        """Create a split-specific segmentation dataset.

        Args:
            results_root: Step #3 result root containing generated Step #2 data.
            split: One of train, validation, or test.
            image_size: Network input width and height.

        Raises:
            ValueError: If split is unsupported.
            FileNotFoundError: If annotations are absent.
        """
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unsupported dataset split")
        annotation_path = results_root / "03_annotations" / "annotations.jsonl"
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        records = [json.loads(line) for line in annotation_path.read_text(encoding="utf-8").splitlines() if line]
        self.records = [
            record for record in records
            if record["split"] == split and record["class"] in {"normal", "scratch"}
        ]
        self.root = results_root
        self.image_size = image_size

    def __len__(self) -> int:
        """Return number of samples in the selected split."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Load and normalize one image/mask pair.

        Only normal and scratch samples are admitted to this AI dataset. Edge, corner,
        and mixed samples are excluded because Step #2 stores one combined defect
        mask and therefore cannot isolate scratch pixels inside mixed samples.
        """
        record = self.records[index]
        image = cv2.imread(str(self.root / record["image"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / record["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError("Dataset image or mask could not be loaded")
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
        if record["class"] != "scratch":
            mask = np.zeros_like(mask)
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor, record["class"]
