"""Aviation-level evaluation for true vs ML vs PIML wind products."""

from __future__ import annotations

import numpy as np

from aviation.flight_paths import DEFAULT_CORRIDORS
from aviation.performance_metrics import (
    corridor_mean_tailwind_series,
    corridor_performance,
    fuel_error_percent,
)


def evaluate_route_fuel(
    true_tailwind_ms: np.ndarray,
    ml_tailwind_ms: np.ndarray,
    piml_tailwind_ms: np.ndarray,
    corridor_name: str = "transcon_us",
) -> dict[str, float]:
    corridor = DEFAULT_CORRIDORS[corridor_name]

    true_perf = corridor_performance(true_tailwind_ms, corridor=corridor)
    ml_perf = corridor_performance(ml_tailwind_ms, corridor=corridor)
    piml_perf = corridor_performance(piml_tailwind_ms, corridor=corridor)

    return {
        "true_fuel_kg": true_perf.fuel_kg,
        "ml_fuel_kg": ml_perf.fuel_kg,
        "piml_fuel_kg": piml_perf.fuel_kg,
        "ml_fuel_error_pct": fuel_error_percent(ml_perf.fuel_kg, true_perf.fuel_kg),
        "piml_fuel_error_pct": fuel_error_percent(piml_perf.fuel_kg, true_perf.fuel_kg),
        "true_ground_speed_ms": true_perf.ground_speed_ms,
        "ml_ground_speed_ms": ml_perf.ground_speed_ms,
        "piml_ground_speed_ms": piml_perf.ground_speed_ms,
    }


def _flight_series_from_tailwind(
    tailwind_series_ms: np.ndarray,
    corridor_name: str,
    aircraft_true_airspeed_ms: float,
) -> dict[str, np.ndarray]:
    corridor = DEFAULT_CORRIDORS[corridor_name]
    perf = [corridor_performance(np.asarray([tw], dtype=np.float32), corridor, aircraft_true_airspeed_ms) for tw in tailwind_series_ms]
    return {
        "ground_speed_ms": np.asarray([p.ground_speed_ms for p in perf], dtype=np.float32),
        "time_seconds": np.asarray([p.time_seconds for p in perf], dtype=np.float32),
        "fuel_kg": np.asarray([p.fuel_kg for p in perf], dtype=np.float32),
    }


def evaluate_wind_products_on_corridor(
    true_uv: np.ndarray,
    ml_uv: np.ndarray,
    piml_uv: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    corridor_name: str = "transcon_us",
    aircraft_true_airspeed_ms: float = 230.0,
    n_route_points: int = 64,
) -> dict[str, float]:
    """Evaluate ML and PIML corridor performance against true winds.

    Args:
        true_uv, ml_uv, piml_uv: [N, 2, H, W] arrays.
    """
    if true_uv.shape != ml_uv.shape or true_uv.shape != piml_uv.shape:
        raise ValueError(f"All wind products must share shape. Got true={true_uv.shape}, ml={ml_uv.shape}, piml={piml_uv.shape}")
    if true_uv.ndim != 4 or true_uv.shape[1] != 2:
        raise ValueError(f"Expected [N,2,H,W] arrays. Got shape {true_uv.shape}")

    corridor = DEFAULT_CORRIDORS[corridor_name]

    true_tailwind = corridor_mean_tailwind_series(
        u_series_ms=true_uv[:, 0],
        v_series_ms=true_uv[:, 1],
        lat_grid=lat_deg,
        lon_grid=lon_deg,
        corridor=corridor,
        n_route_points=n_route_points,
    )
    ml_tailwind = corridor_mean_tailwind_series(
        u_series_ms=ml_uv[:, 0],
        v_series_ms=ml_uv[:, 1],
        lat_grid=lat_deg,
        lon_grid=lon_deg,
        corridor=corridor,
        n_route_points=n_route_points,
    )
    piml_tailwind = corridor_mean_tailwind_series(
        u_series_ms=piml_uv[:, 0],
        v_series_ms=piml_uv[:, 1],
        lat_grid=lat_deg,
        lon_grid=lon_deg,
        corridor=corridor,
        n_route_points=n_route_points,
    )

    true_series = _flight_series_from_tailwind(true_tailwind, corridor_name, aircraft_true_airspeed_ms)
    ml_series = _flight_series_from_tailwind(ml_tailwind, corridor_name, aircraft_true_airspeed_ms)
    piml_series = _flight_series_from_tailwind(piml_tailwind, corridor_name, aircraft_true_airspeed_ms)

    ml_fuel_err_pct = 100.0 * (ml_series["fuel_kg"] - true_series["fuel_kg"]) / np.clip(true_series["fuel_kg"], 1e-6, None)
    piml_fuel_err_pct = 100.0 * (piml_series["fuel_kg"] - true_series["fuel_kg"]) / np.clip(true_series["fuel_kg"], 1e-6, None)

    return {
        "n_flights": float(true_uv.shape[0]),
        "true_tailwind_mean_ms": float(np.mean(true_tailwind)),
        "ml_tailwind_mean_ms": float(np.mean(ml_tailwind)),
        "piml_tailwind_mean_ms": float(np.mean(piml_tailwind)),
        "true_ground_speed_mean_ms": float(np.mean(true_series["ground_speed_ms"])),
        "ml_ground_speed_mean_ms": float(np.mean(ml_series["ground_speed_ms"])),
        "piml_ground_speed_mean_ms": float(np.mean(piml_series["ground_speed_ms"])),
        "true_fuel_mean_kg": float(np.mean(true_series["fuel_kg"])),
        "ml_fuel_mean_kg": float(np.mean(ml_series["fuel_kg"])),
        "piml_fuel_mean_kg": float(np.mean(piml_series["fuel_kg"])),
        "ml_fuel_error_mean_pct": float(np.mean(ml_fuel_err_pct)),
        "piml_fuel_error_mean_pct": float(np.mean(piml_fuel_err_pct)),
        "ml_fuel_error_std_pct": float(np.std(ml_fuel_err_pct)),
        "piml_fuel_error_std_pct": float(np.std(piml_fuel_err_pct)),
        "true_time_std_seconds": float(np.std(true_series["time_seconds"])),
        "ml_time_std_seconds": float(np.std(ml_series["time_seconds"])),
        "piml_time_std_seconds": float(np.std(piml_series["time_seconds"])),
    }
