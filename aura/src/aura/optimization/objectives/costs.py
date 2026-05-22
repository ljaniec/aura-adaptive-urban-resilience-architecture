"""Multi-objective scoring functions."""

from __future__ import annotations


def weighted_objective(metrics: dict, weights: dict) -> float:
    """Compute weighted objective value from metric dictionary."""
    return (
        weights["indoor_temperature_error"] * float(metrics["indoor_temperature_error"])
        + weights["drag_force"] * float(metrics["drag_force"])
        + weights["vortex_intensity"] * float(metrics["vortex_intensity"])
        + weights["pedestrian_discomfort"] * float(metrics["pedestrian_wind_comfort"])
        + weights["energy_cost"] * float(metrics["energy_cost"])
    )
