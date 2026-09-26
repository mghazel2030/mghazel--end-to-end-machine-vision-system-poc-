"""Step #3 PyTorch model and metric tests."""
import torch

from vision_poc.metrics import binary_segmentation_metrics
from vision_poc.model import TinyUNet


def test_unet_preserves_spatial_dimensions() -> None:
    """Verify binary segmentation logits match the input spatial resolution."""
    model = TinyUNet(base_channels=4)
    x = torch.zeros((2, 1, 64, 80))
    assert model(x).shape == (2, 1, 64, 80)


def test_binary_metrics_perfect_prediction() -> None:
    """Verify metric implementation on a perfect binary prediction."""
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    metrics = binary_segmentation_metrics(logits, target)
    assert metrics["dice"] > 0.999
    assert metrics["iou"] > 0.999
