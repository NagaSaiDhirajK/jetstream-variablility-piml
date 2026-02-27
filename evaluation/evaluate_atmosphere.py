"""Atmospheric evaluation metrics for wind prediction + physics consistency."""

from __future__ import annotations

import torch

from physics.geostrophic import divergence_residual_loss
from physics.geostrophic import geostrophic_residual_loss
from physics.geostrophic import vorticity_residual_loss
from physics.thermal_wind import thermal_wind_residual_loss


def wind_rmse(pred_uv: torch.Tensor, true_uv: torch.Tensor) -> dict[str, float]:
    rmse_u = torch.sqrt(torch.mean((pred_uv[:, 0] - true_uv[:, 0]) ** 2))
    rmse_v = torch.sqrt(torch.mean((pred_uv[:, 1] - true_uv[:, 1]) ** 2))
    return {"rmse_u": float(rmse_u.detach().cpu()), "rmse_v": float(rmse_v.detach().cpu())}


@torch.no_grad()
def atmospheric_metrics(
    pred_uv: torch.Tensor,
    true_uv: torch.Tensor,
    geopotential: torch.Tensor,
    temperature_mid: torch.Tensor,
    u_bottom: torch.Tensor,
    v_bottom: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    p_top_hpa: float = 200.0,
    p_bottom_hpa: float = 350.0,
) -> dict[str, float]:
    """Evaluate wind RMSE and physics-violation magnitudes on a dataset subset."""
    metrics = wind_rmse(pred_uv=pred_uv, true_uv=true_uv)
    geo = geostrophic_residual_loss(
        u_pred=pred_uv[:, 0],
        v_pred=pred_uv[:, 1],
        geopotential=geopotential,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    tw = thermal_wind_residual_loss(
        u_pred_top=pred_uv[:, 0],
        v_pred_top=pred_uv[:, 1],
        u_ref_bottom=u_bottom,
        v_ref_bottom=v_bottom,
        temperature_mid=temperature_mid,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        p_top_hpa=p_top_hpa,
        p_bottom_hpa=p_bottom_hpa,
    )
    div = divergence_residual_loss(
        u_pred=pred_uv[:, 0],
        v_pred=pred_uv[:, 1],
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    vort = vorticity_residual_loss(
        u_pred=pred_uv[:, 0],
        v_pred=pred_uv[:, 1],
        geopotential=geopotential,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
    )
    metrics["geo_violation_mse"] = float(geo.detach().cpu())
    metrics["thermal_wind_violation_mse"] = float(tw.detach().cpu())
    metrics["divergence_violation_mse"] = float(div.detach().cpu())
    metrics["vorticity_violation_mse"] = float(vort.detach().cpu())
    return metrics
