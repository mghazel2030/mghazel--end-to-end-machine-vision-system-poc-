"""Training and evaluation loop for the compact PyTorch U-Net.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import binary_segmentation_metrics
from .model import TinyUNet
from .torch_data import ScratchSegmentationDataset


def _epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device,
           optimizer: torch.optim.Optimizer | None = None) -> tuple[float, dict[str, float]]:
    """Run one training or evaluation epoch and aggregate metrics."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    aggregate = {"precision": 0.0, "recall": 0.0, "dice": 0.0, "iou": 0.0}
    batches = 0
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        current = binary_segmentation_metrics(logits.detach(), masks)
        for name in aggregate:
            aggregate[name] += current[name]
        batches += 1
    divisor = max(batches, 1)
    return total_loss / divisor, {name: value / divisor for name, value in aggregate.items()}


def train_model(cfg: dict[str, Any], dataset_root: Path, output: Path) -> dict[str, Any]:
    """Train, validate, test, and save a compact scratch-segmentation U-Net.

    Args:
        cfg: Project configuration including training parameters.
        dataset_root: Generated synthetic dataset root.
        output: Step #3 AI output directory.

    Returns:
        Training history, held-out test metrics, and model path.
    """
    tcfg = cfg["training"]
    torch.manual_seed(int(tcfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and tcfg["device"] == "auto" else "cpu")
    image_size = (int(tcfg["input_width_px"]), int(tcfg["input_height_px"]))
    train_ds = ScratchSegmentationDataset(dataset_root, "train", image_size)
    val_ds = ScratchSegmentationDataset(dataset_root, "validation", image_size)
    test_ds = ScratchSegmentationDataset(dataset_root, "test", image_size)
    generator = torch.Generator().manual_seed(int(tcfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=int(tcfg["batch_size"]), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(tcfg["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(tcfg["batch_size"]), shuffle=False)
    model = TinyUNet(int(tcfg["base_channels"])).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(tcfg["positive_weight"])], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["learning_rate"]))
    history = []
    best_loss = float("inf")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "best_scratch_unet.pt"
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        train_loss, train_metrics = _epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_metrics = _epoch(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "train": train_metrics, "validation": val_metrics})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_loss, test_metrics = _epoch(model, test_loader, loss_fn, device)
    return {"device": str(device), "history": history, "test_loss": test_loss,
            "test_metrics": test_metrics, "model_path": str(model_path)}
