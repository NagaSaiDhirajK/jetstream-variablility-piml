"""Thermal-wind diagnostics and residual losses.

For pressure levels p_top and p_bottom:
    du/dlnp = -(R/f) * dT/dy
    dv/dlnp =  (R/f) * dT/dx
"""

from __future__ import annotations

import math

import torch

from physics.geostrophic import EARTH_RADIUS_M, coriolis_parameter

R_DRY_AIR = 287.05


def _validate_pressure_pair(p_top_hpa: float, p_bottom_hpa: float) -> None:
    if p_top_hpa <= 0.0 or p_bottom_hpa <= 0.0:
        raise ValueError(
            f"Pressure levels must be positive, got p_top_hpa={p_top_hpa}, p_bottom_hpa={p_bottom_hpa}."
        )
    if p_bottom_hpa <= p_top_hpa:
        raise ValueError(
            f"Expected bottom pressure > top pressure, got p_top_hpa={p_top_hpa}, p_bottom_hpa={p_bottom_hpa}."
        )


def thermal_wind_shear(
    temperature_mid: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    p_top_hpa: float = 200.0,
    p_bottom_hpa: float = 350.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute expected wind shear from horizontal temperature gradients.

    Returns (du, dv) over ln(p_bottom / p_top).
    """
    _validate_pressure_pair(p_top_hpa=p_top_hpa, p_bottom_hpa=p_bottom_hpa)
    lat_rad = torch.deg2rad(lat_deg)
    lon_rad = torch.deg2rad(lon_deg)

    dlat = torch.gradient(lat_rad, spacing=1.0)[0]
    dlon = torch.gradient(lon_rad, spacing=1.0)[0]
    dy = EARTH_RADIUS_M * dlat
    dx = EARTH_RADIUS_M * torch.cos(lat_rad)[:, None] * dlon[None, :]

    dt_dy = torch.gradient(temperature_mid, dim=1)[0] / dy[None, :, None]
    dt_dx = torch.gradient(temperature_mid, dim=2)[0] / dx[None, :, :]

    f = coriolis_parameter(lat_deg)[None, :, None]
    dlnp = math.log(p_bottom_hpa / p_top_hpa)

    du = (-(R_DRY_AIR / f) * dt_dy) * dlnp
    dv = ((R_DRY_AIR / f) * dt_dx) * dlnp
    return du, dv


def thermal_wind_residual_loss(
    u_pred_top: torch.Tensor,
    v_pred_top: torch.Tensor,
    u_ref_bottom: torch.Tensor,
    v_ref_bottom: torch.Tensor,
    temperature_mid: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
    p_top_hpa: float = 200.0,
    p_bottom_hpa: float = 350.0,
) -> torch.Tensor:
    """L_tw = ||predicted_shear - thermal_wind_shear||^2.

    Phase-1 approximates predicted shear using (predicted top wind - provided bottom wind).
    """
    _validate_pressure_pair(p_top_hpa=p_top_hpa, p_bottom_hpa=p_bottom_hpa)
    target_du, target_dv = thermal_wind_shear(
        temperature_mid=temperature_mid,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        p_top_hpa=p_top_hpa,
        p_bottom_hpa=p_bottom_hpa,
    )

    pred_du = u_pred_top - u_ref_bottom
    pred_dv = v_pred_top - v_ref_bottom
    return torch.mean((pred_du - target_du) ** 2 + (pred_dv - target_dv) ** 2)
