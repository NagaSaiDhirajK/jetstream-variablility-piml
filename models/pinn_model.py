"""Physics-informed model wrapper for cruise-level wind surrogate."""

from __future__ import annotations

import torch
from torch import nn

from models.base_model import BaselineWindCNN


class PhysicsInformedWindModel(nn.Module):
    """Baseline model that can be trained with additional physics residual losses."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_res_blocks: int = 4,
        dropout: float = 0.0,
        use_bottom_residual: bool = True,
    ) -> None:
        super().__init__()
        # Backbone predicts either top-level wind directly or a residual shear field.
        self.backbone = BaselineWindCNN(
            in_channels=in_channels,
            out_channels=2,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )
        self.use_bottom_residual = use_bottom_residual

    def forward(
        self,
        x: torch.Tensor,
        u_bottom: torch.Tensor | None = None,
        v_bottom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self.backbone(x)
        if not self.use_bottom_residual or u_bottom is None or v_bottom is None:
            return pred
        bottom_uv = torch.stack([u_bottom, v_bottom], dim=1).to(dtype=pred.dtype, device=pred.device)
        return pred + bottom_uv
