"""Residual CNN architecture shared by training and inference.

Import requires torch (optional dependency); aidetect.ml_detector guards it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter


def residual_tensor(img: Image.Image) -> torch.Tensor:
    """High-pass residual input: image minus its Gaussian blur, scaled."""
    rgb = np.asarray(img, dtype=np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    res = (rgb - blur) / 25.0
    return torch.from_numpy(res.transpose(2, 0, 1))


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)
