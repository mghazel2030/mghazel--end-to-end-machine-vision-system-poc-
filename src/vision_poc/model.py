"""Compact PyTorch U-Net for scratch segmentation.

Author: mghazel
Submitted to: Ascension Automation Solutions Ltd.
Version: 2026-09-25
"""
import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two convolution-normalization-activation operations."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize a convolution block."""
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block to a tensor."""
        return self.block(x)


class TinyUNet(nn.Module):
    """Small U-Net suitable for CPU PoC training and binary segmentation."""

    def __init__(self, base_channels: int = 16) -> None:
        """Initialize encoder, bottleneck, decoder, and segmentation head."""
        super().__init__()
        b = base_channels
        self.enc1 = ConvBlock(1, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(b * 2, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return unnormalized per-pixel scratch logits."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        z = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat((self.up2(z), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
        return self.head(d1)
