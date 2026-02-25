"""Representative aviation corridors and utilities."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlightCorridor:
    name: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    distance_m: float


DEFAULT_CORRIDORS = {
    "midwest_to_east": FlightCorridor(
        name="Midwest to East Coast",
        start_lat=41.9742,
        start_lon=-87.9073,
        end_lat=40.6413,
        end_lon=-73.7781,
        distance_m=1_200_000.0,
    ),
    "transcon_us": FlightCorridor(
        name="Transcontinental US",
        start_lat=33.9416,
        start_lon=-118.4085,
        end_lat=40.6413,
        end_lon=-73.7781,
        distance_m=3_980_000.0,
    ),
    "north_atlantic": FlightCorridor(
        name="North Atlantic",
        start_lat=40.6413,
        start_lon=-73.7781,
        end_lat=51.4700,
        end_lon=-0.4543,
        distance_m=5_550_000.0,
    ),
}


def wrap_longitudes(lon_deg: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180)."""
    return ((lon_deg + 180.0) % 360.0) - 180.0


def route_points(corridor: FlightCorridor, n_points: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return equally spaced lat/lon points along a corridor."""
    lats = np.linspace(corridor.start_lat, corridor.end_lat, n_points, dtype=np.float32)
    lons = np.linspace(corridor.start_lon, corridor.end_lon, n_points, dtype=np.float32)
    return lats, wrap_longitudes(lons)


def track_unit_vector_xy(corridor: FlightCorridor) -> tuple[float, float]:
    """Approximate local track direction as (east, north) unit vector."""
    mean_lat_rad = np.deg2rad(0.5 * (corridor.start_lat + corridor.end_lat))
    dx = (corridor.end_lon - corridor.start_lon) * np.cos(mean_lat_rad)
    dy = corridor.end_lat - corridor.start_lat
    norm = float(np.hypot(dx, dy))
    if norm < 1e-8:
        return (1.0, 0.0)
    return (float(dx / norm), float(dy / norm))
