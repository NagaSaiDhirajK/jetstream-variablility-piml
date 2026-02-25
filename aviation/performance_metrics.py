"""Aviation performance metrics derived from predicted wind fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aviation.flight_paths import FlightCorridor, route_points, track_unit_vector_xy, wrap_longitudes
from aviation.fuel_model import fuel_burn_from_time_seconds


@dataclass
class CorridorPerformance:
    mean_tailwind_ms: float
    ground_speed_ms: float
    time_seconds: float
    fuel_kg: float


def project_wind_to_track(u_ms: np.ndarray, v_ms: np.ndarray, track_unit_xy: tuple[float, float]) -> np.ndarray:
    """Project horizontal wind vectors onto route track direction."""
    tx, ty = track_unit_xy
    return u_ms * tx + v_ms * ty


def _nearest_lat_lon_indices(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    route_lats: np.ndarray,
    route_lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lat_diff = np.abs(lat_grid[:, None] - route_lats[None, :])
    lat_idx = np.argmin(lat_diff, axis=0)

    lon_wrapped = wrap_longitudes(lon_grid)
    route_lon_wrapped = wrap_longitudes(route_lons)
    lon_diff = np.abs(lon_wrapped[:, None] - route_lon_wrapped[None, :])
    lon_diff = np.minimum(lon_diff, 360.0 - lon_diff)
    lon_idx = np.argmin(lon_diff, axis=0)
    return lat_idx, lon_idx


def corridor_tailwind_from_grid(
    u_grid_ms: np.ndarray,
    v_grid_ms: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    corridor: FlightCorridor,
    n_route_points: int = 64,
) -> np.ndarray:
    """Sample grid winds along a corridor and project onto route direction."""
    route_lats, route_lons = route_points(corridor, n_points=n_route_points)
    lat_idx, lon_idx = _nearest_lat_lon_indices(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        route_lats=route_lats,
        route_lons=route_lons,
    )
    u_route = u_grid_ms[lat_idx, lon_idx]
    v_route = v_grid_ms[lat_idx, lon_idx]
    return project_wind_to_track(u_route, v_route, track_unit_vector_xy(corridor))


def corridor_mean_tailwind_series(
    u_series_ms: np.ndarray,
    v_series_ms: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    corridor: FlightCorridor,
    n_route_points: int = 64,
) -> np.ndarray:
    """Compute mean corridor tailwind for each time sample."""
    if u_series_ms.shape != v_series_ms.shape:
        raise ValueError(f"u_series shape {u_series_ms.shape} must match v_series shape {v_series_ms.shape}")

    n = u_series_ms.shape[0]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        tw = corridor_tailwind_from_grid(
            u_grid_ms=u_series_ms[i],
            v_grid_ms=v_series_ms[i],
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            corridor=corridor,
            n_route_points=n_route_points,
        )
        out[i] = float(np.mean(tw))
    return out


def corridor_performance(
    tailwind_ms: np.ndarray,
    corridor: FlightCorridor,
    aircraft_true_airspeed_ms: float = 230.0,
) -> CorridorPerformance:
    """Compute route-level speed/time/fuel from tailwind samples."""
    mean_tailwind = float(np.mean(tailwind_ms))
    ground_speed = max(50.0, aircraft_true_airspeed_ms + mean_tailwind)
    time_seconds = corridor.distance_m / ground_speed
    fuel_kg = fuel_burn_from_time_seconds(time_seconds)
    return CorridorPerformance(
        mean_tailwind_ms=mean_tailwind,
        ground_speed_ms=ground_speed,
        time_seconds=time_seconds,
        fuel_kg=fuel_kg,
    )


def fuel_error_percent(pred_fuel_kg: float, true_fuel_kg: float) -> float:
    if true_fuel_kg <= 0:
        return 0.0
    return 100.0 * (pred_fuel_kg - true_fuel_kg) / true_fuel_kg
