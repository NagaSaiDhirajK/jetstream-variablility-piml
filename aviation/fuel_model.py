"""Simple fuel burn sensitivity model for corridor-level comparisons."""

from __future__ import annotations


def fuel_burn_from_time_seconds(time_seconds: float, fuel_flow_kg_per_s: float = 2.5) -> float:
    """Approximate fuel burn (kg) as proportional to cruise time."""
    return time_seconds * fuel_flow_kg_per_s
