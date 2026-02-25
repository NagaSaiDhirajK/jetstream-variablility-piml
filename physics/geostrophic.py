"""Geostrophic diagnostics and residual losses.

Geostrophic balance:
    u_g = -(1/f) * dPhi/dy
    v_g =  (1/f) * dPhi/dx
where f = 2 * Omega * sin(latitude).
"""

from __future__ import annotations

import math

import torch

OMEGA = 7.2921159e-5
EARTH_RADIUS_M = 6_371_000.0


def coriolis_parameter(lat_deg: torch.Tensor) -> torch.Tensor:
    """Compute Coriolis parameter f for latitude in degrees."""
    lat_rad = torch.deg2rad(lat_deg)
    f = 2.0 * OMEGA * torch.sin(lat_rad)
    return torch.where(torch.abs(f) < 1e-5, torch.sign(f) * 1e-5 + (f == 0.0) * 1e-5, f)


def _grid_spacing_m(lat_deg: torch.Tensor, lon_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return meridional and zonal grid spacing in meters.

    dy is 1D over latitude, dx is 2D over [lat, lon] because longitudinal spacing
    scales with cos(latitude).
    """
    lat_rad = torch.deg2rad(lat_deg)
    lon_rad = torch.deg2rad(lon_deg)

    dlat = torch.gradient(lat_rad, spacing=1.0)[0]
    dlon = torch.gradient(lon_rad, spacing=1.0)[0]

    dy = EARTH_RADIUS_M * dlat
    dx_1d = EARTH_RADIUS_M * torch.cos(lat_rad)[:, None] * dlon[None, :]
    return dy, dx_1d


def _safe_central_gradient(field: torch.Tensor, spacing: torch.Tensor, dim: int) -> torch.Tensor:
    """Central-difference gradient with variable spacing along one dimension."""
    grad = torch.gradient(field, dim=dim)[0]

    view_shape = [1] * field.ndim
    view_shape[dim] = spacing.shape[0]
    return grad / spacing.view(*view_shape)


def geostrophic_wind(
    geopotential: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute geostrophic wind from geopotential.

    Args:
        geopotential: [B, H, W] geopotential (m^2/s^2)
        lat_deg: [H] latitude array
        lon_deg: [W] longitude array
    """
    dy, dx = _grid_spacing_m(lat_deg, lon_deg)

    dphi_dy = _safe_central_gradient(geopotential, dy, dim=1)
    dphi_dx = torch.gradient(geopotential, dim=2)[0] / dx[None, :, :]

    f = coriolis_parameter(lat_deg)[None, :, None]
    u_g = -(1.0 / f) * dphi_dy
    v_g = (1.0 / f) * dphi_dx
    return u_g, v_g


def geostrophic_residual_loss(
    u_pred: torch.Tensor,
    v_pred: torch.Tensor,
    geopotential: torch.Tensor,
    lat_deg: torch.Tensor,
    lon_deg: torch.Tensor,
) -> torch.Tensor:
    """L_geo = ||u_pred - u_g||^2 + ||v_pred - v_g||^2."""
    u_g, v_g = geostrophic_wind(geopotential=geopotential, lat_deg=lat_deg, lon_deg=lon_deg)
    return torch.mean((u_pred - u_g) ** 2 + (v_pred - v_g) ** 2)
