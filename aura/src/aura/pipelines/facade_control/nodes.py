"""Nodes for adaptive facade control."""

from __future__ import annotations

import numpy as np

from aura.controllers.rule_based import compute_rule_based_panels


def compute_facade_state(
    external_flow_state: dict,
    facade: dict,
    controllers: dict,
    thermal: dict,
) -> dict:
    """Compute louver angles and opening ratios for the current state."""
    pressure = external_flow_state["pressure_field"]

    indoor_temp = float(thermal["initial_indoor_temp"])
    outdoor_temp = float(thermal["outdoor_temp"])
    wind_direction = float(controllers.get("wind_direction_override_deg", 0.0))

    mode = controllers.get("mode", "rule_based")
    if mode == "rule_based":
        panel_state = compute_rule_based_panels(
            external_pressure=pressure,
            indoor_temperature=indoor_temp,
            outdoor_temperature=outdoor_temp,
            wind_direction_deg=wind_direction,
            facade_cfg=facade,
        )
    elif mode == "static":
        n = int(facade["num_panels"])
        static_cfg = controllers.get("static", {})
        fixed_angle = float(static_cfg.get("fixed_angle", 0.0))
        fixed_opening = float(static_cfg.get("fixed_opening", 0.35))
        panel_state = {
            "panel_angle": [fixed_angle] * n,
            "panel_opening": [fixed_opening] * n,
            "mode": "static",
        }
    else:
        raise ValueError(
            f"Unsupported controller mode '{mode}'. "
            "Supported modes: 'rule_based', 'static'."
        )

    panel_state["permeability"] = np.clip(
        np.array(panel_state["panel_opening"], dtype=float),
        float(facade["opening_min"]),
        float(facade["opening_max"]),
    ).tolist()
    return panel_state
