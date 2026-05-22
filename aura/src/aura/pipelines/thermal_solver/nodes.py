"""Nodes for thermal advection-diffusion and buoyancy approximation."""

from __future__ import annotations

import numpy as np

from aura.physics.simplified import advect_scalar, laplacian


def run_thermal_solver(
    indoor_flow_state: dict,
    facade_state: dict,
    simulation: dict,
    domain: dict,
    thermal: dict,
    building: dict,
) -> dict:
    """Advance a coarse advection-diffusion temperature field."""
    del facade_state, building

    u = indoor_flow_state["indoor_velocity"]["u"]
    v = indoor_flow_state["indoor_velocity"]["v"]
    u = np.clip(np.nan_to_num(u, nan=0.0), -5.0, 5.0)
    v = np.clip(np.nan_to_num(v, nan=0.0), -5.0, 5.0)

    nx = int(domain["nx"])
    ny = int(domain["ny"])
    dx = float(domain["dx"])
    dy = float(domain.get("dy", dx))
    dt = float(simulation["dt"])
    steps = int(simulation["thermal_steps"])

    alpha = float(thermal["alpha"])
    buoyancy_beta = float(thermal["buoyancy_beta"])
    gravity = float(thermal["gravity"])

    T = np.full((nx, ny), float(thermal["initial_indoor_temp"]), dtype=float)
    T_out = float(thermal["outdoor_temp"])

    for _ in range(steps):
        T[0, :] = T_out
        T_adv = advect_scalar(T, u=u, v=v, dt=dt, dx=dx, dy=dy)
        T = T_adv + dt * alpha * laplacian(T, dx=dx, dy=dy)
        T = np.clip(np.nan_to_num(T, nan=T_out), T_out - 40.0, T_out + 60.0)

        # Buoyancy proxy: warm air induces upward motion.
        v += dt * buoyancy_beta * (T - T_out) * gravity
        v = np.clip(np.nan_to_num(v, nan=0.0), -5.0, 5.0)

    heat_flux = -alpha * np.gradient(T, dx, axis=0)
    return {
        "temperature_field": T,
        "heat_flux": heat_flux,
        "mean_indoor_temp": float(np.mean(T)),
    }
