"""Physics-informed model wrapper for cruise-level wind surrogate."""

from __future__ import annotations

import torch
from torch import nn

from models.base_model import BaselineWindCNN


class PhysicsInformedWindModel(nn.Module):
    """Baseline model that can be trained with additional physics residual losses."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        # Output is [u_pred, v_pred] at the target pressure level for phase-1.
        self.backbone = BaselineWindCNN(
            in_channels=in_channels,
            out_channels=2,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
