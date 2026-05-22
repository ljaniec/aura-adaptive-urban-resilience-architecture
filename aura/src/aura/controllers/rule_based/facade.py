"""Rule-based facade controller for rapid prototyping."""

from __future__ import annotations

import numpy as np


def compute_rule_based_panels(
    external_pressure: np.ndarray,
    indoor_temperature: float,
    outdoor_temperature: float,
    wind_direction_deg: float,
    facade_cfg: dict,
) -> dict:
    """Generate panel angles/openings from pressure and thermal gradients."""
    n = int(facade_cfg["num_panels"])
    angle_min = float(facade_cfg["angle_min"])
    angle_max = float(facade_cfg["angle_max"])
    opening_min = float(facade_cfg["opening_min"])
    opening_max = float(facade_cfg["opening_max"])

    pressure_line = external_pressure[:, external_pressure.shape[1] // 2]
    sampled = np.interp(
        np.linspace(0, pressure_line.shape[0] - 1, n),
        np.arange(pressure_line.shape[0]),
        pressure_line,
    )
    sampled_norm = (sampled - sampled.min()) / (np.ptp(sampled) + 1e-9)

    thermal_delta = indoor_temperature - outdoor_temperature
    openness = opening_min + (opening_max - opening_min) * (0.4 + 0.6 * sampled_norm)
    if thermal_delta > 0:
        openness = np.clip(openness + 0.15, opening_min, opening_max)

    wind_factor = np.cos(np.deg2rad(wind_direction_deg))
    center_bias = np.linspace(-1.0, 1.0, n)
    angles = wind_factor * center_bias * angle_max
    angles = np.clip(angles, angle_min, angle_max)

    return {
        "panel_angle": angles.tolist(),
        "panel_opening": openness.tolist(),
        "mode": "rule_based",
    }
