"""Nodes for simulation KPI and cost calculations."""

from __future__ import annotations

import numpy as np

from aura.optimization.objectives import weighted_objective


def compute_simulation_metrics(
    external_flow_state: dict,
    indoor_flow_state: dict,
    thermal_state: dict,
    facade_state: dict,
    metrics: dict,
) -> dict:
    """Compute evaluation metrics and weighted objective."""
    pressure = external_flow_state["pressure_field"]
    vorticity = external_flow_state["vorticity_field"]

    indoor_temp = float(thermal_state["mean_indoor_temp"])
    target_temp = float(metrics["target_indoor_temp"])
    indoor_temperature_error = abs(indoor_temp - target_temp)

    drag_force = float(np.mean(np.abs(pressure)))
    vortex_intensity = float(np.mean(np.abs(vorticity)))
    pedestrian_wind_comfort = float(np.percentile(np.abs(vorticity), 90))
    air_exchange_rate = float(indoor_flow_state["air_exchange_rate"])
    pressure_drop = float(np.max(pressure) - np.min(pressure))
    energy_cost = float(
        np.mean(np.array(facade_state["panel_opening"], dtype=float) ** 2)
    )

    report = {
        "mean_indoor_temp": indoor_temp,
        "air_exchange_rate": air_exchange_rate,
        "pressure_drop": pressure_drop,
        "drag_force": drag_force,
        "vortex_intensity": vortex_intensity,
        "pedestrian_wind_comfort": pedestrian_wind_comfort,
        "energy_cost": energy_cost,
        "indoor_temperature_error": indoor_temperature_error,
    }
    report["objective"] = weighted_objective(report, metrics["weights"])
    return report
