"""Base neural network model for wind-field surrogate prediction."""

from __future__ import annotations

import torch
from torch import nn


class BaselineWindCNN(nn.Module):
    """Small CNN for spatial regression of upper-level wind fields.

    Inputs are gridded atmospheric predictors (for example T, geopotential, prior winds),
    and outputs are predicted u/v wind components at target pressure levels.
    """

    def __init__(self, in_channels: int, out_channels: int = 2, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
