"""Nodes for indoor ventilation simulation."""

from __future__ import annotations

import numpy as np

from aura.physics.simplified import compute_vorticity, pressure_smoothing


def run_internal_cfd(
    external_flow_state: dict,
    facade_state: dict,
    simulation: dict,
    domain: dict,
    fluid: dict,
    building: dict,
    facade: dict,
) -> dict:
    """Approximate indoor airflow as a damped continuation of external flow."""
    del building, facade

    u_ext = external_flow_state["velocity_field"]["u"]
    v_ext = external_flow_state["velocity_field"]["v"]
    geometry = external_flow_state["geometry_mask"]

    nx = int(domain["nx"])
    ny = int(domain["ny"])
    dx = float(domain["dx"])
    dy = float(domain.get("dy", dx))
    dt = float(simulation["dt"])
    steps = int(simulation["internal_steps"])
    nu = float(fluid["mu"]) / float(fluid["rho"])

    u = np.zeros((nx, ny), dtype=float)
    v = np.zeros((nx, ny), dtype=float)
    p = np.zeros((nx, ny), dtype=float)

    panel_opening = np.mean(np.array(facade_state["panel_opening"], dtype=float))

    for _ in range(steps):
        p = pressure_smoothing(p, iterations=1)

        # Interior inherits a scaled version of nearby external flow.
        u = 0.96 * u + 0.04 * (panel_opening * u_ext)
        v = 0.96 * v + 0.04 * (panel_opening * v_ext)

        lap_u = (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / (
            dx**2
        ) + (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dy**2)
        lap_v = (np.roll(v, -1, axis=0) - 2.0 * v + np.roll(v, 1, axis=0)) / (
            dx**2
        ) + (np.roll(v, -1, axis=1) - 2.0 * v + np.roll(v, 1, axis=1)) / (dy**2)
        u = u + dt * nu * lap_u
        v = v + dt * nu * lap_v

        solid = geometry == 1
        u[solid] = 0.0
        v[solid] = 0.0

        p = p + 0.03 * (u**2 + v**2)

    indoor_speed = np.sqrt(u**2 + v**2)
    air_exchange_rate = float(np.mean(indoor_speed))
    return {
        "indoor_velocity": {"u": u, "v": v},
        "indoor_pressure": p,
        "air_exchange_rate": air_exchange_rate,
        "indoor_vorticity": compute_vorticity(u=u, v=v, dx=dx, dy=dy),
    }
