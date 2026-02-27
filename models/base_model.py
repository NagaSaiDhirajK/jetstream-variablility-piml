"""Base neural network model for wind-field surrogate prediction."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Two-layer residual block with GroupNorm for small-batch stability."""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class BaselineWindCNN(nn.Module):
    """Small CNN for spatial regression of upper-level wind fields.

    Inputs are gridded atmospheric predictors (for example T, geopotential, prior winds),
    and outputs are predicted u/v wind components at target pressure levels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualConvBlock(channels=hidden_channels, dropout=dropout) for _ in range(num_res_blocks)]
        )
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
