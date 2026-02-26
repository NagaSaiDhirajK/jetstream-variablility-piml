"""Loss composition for data and physics-informed training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from physics.geostrophic import divergence_residual_loss
from physics.geostrophic import geostrophic_residual_loss
from physics.thermal_wind import thermal_wind_residual_loss


@dataclass
class LossWeights:
    lambda_geo: float = 0.1
    lambda_tw: float = 0.1
    lambda_div: float = 0.0


@dataclass
class LossBreakdown:
    total: torch.Tensor
    data: torch.Tensor
    geostrophic: torch.Tensor
    thermal_wind: torch.Tensor
    divergence: torch.Tensor


def compute_total_loss(
    pred_uv: torch.Tensor,
    true_uv: torch.Tensor,
    geopotential: torch.Tensor,
    temperature_mid: torch.Tensor,
    u_ref_bottom: torch.Tensor,
    v_ref_bottom: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    weights: LossWeights,
) -> LossBreakdown:
    """L_total = L_data + lambda_geo * L_geo + lambda_tw * L_tw + lambda_div * L_div."""
    u_pred = pred_uv[:, 0]
    v_pred = pred_uv[:, 1]

    data_loss = F.mse_loss(pred_uv, true_uv)
    geo_loss = geostrophic_residual_loss(
        u_pred=u_pred,
        v_pred=v_pred,
        geopotential=geopotential,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    tw_loss = thermal_wind_residual_loss(
        u_pred_top=u_pred,
        v_pred_top=v_pred,
        u_ref_bottom=u_ref_bottom,
        v_ref_bottom=v_ref_bottom,
        temperature_mid=temperature_mid,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    div_loss = divergence_residual_loss(
        u_pred=u_pred,
        v_pred=v_pred,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )

    total = (
        data_loss
        + (weights.lambda_geo * geo_loss)
        + (weights.lambda_tw * tw_loss)
        + (weights.lambda_div * div_loss)
    )
    return LossBreakdown(
        total=total,
        data=data_loss,
        geostrophic=geo_loss,
        thermal_wind=tw_loss,
        divergence=div_loss,
    )
