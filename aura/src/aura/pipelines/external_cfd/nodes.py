"""Nodes for external airflow simulation (Level-1 simplified model)."""

from __future__ import annotations

import numpy as np

from aura.physics.geometry import build_geometry_mask
from aura.physics.lbm import run_lbm_d2q9
from aura.physics.simplified import compute_vorticity, pressure_smoothing


def run_external_cfd(
    wind_inlet: dict,
    simulation: dict,
    domain: dict,
    fluid: dict,
    building: dict,
    facade: dict,
    external_cfd: dict,
) -> dict:
    """Run external airflow using selected backend (simplified or LBM)."""
    nx = int(domain["nx"])
    ny = int(domain["ny"])
    dx = float(domain["dx"])
    dy = float(domain.get("dy", dx))
    dt = float(simulation["dt"])
    steps = int(simulation["steps"])
    nu = float(fluid["mu"]) / float(fluid["rho"])

    geometry_mask = build_geometry_mask(nx=nx, ny=ny, building=building, facade=facade)
    u = np.zeros((nx, ny), dtype=float)
    v = np.zeros((nx, ny), dtype=float)
    p = np.zeros((nx, ny), dtype=float)

    u_inlet = wind_inlet["u_inlet"]
    v_inlet = wind_inlet["v_inlet"]

    solver = external_cfd.get("solver", "lbm_d2q9")
    if solver == "lbm_d2q9":
        lbm_cfg = dict(external_cfd.get("lbm", {}))
        lbm_cfg.setdefault("steps", steps)
        return run_lbm_d2q9(
            u_inlet=u_inlet,
            v_inlet=v_inlet,
            geometry_mask=geometry_mask,
            lbm_cfg=lbm_cfg,
            dx=dx,
            dy=dy,
        )

    for t in range(steps):
        u[0, :] = u_inlet[t, :]
        v[0, :] = v_inlet[t, :]

        p = pressure_smoothing(p, iterations=2)

        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dx)
        dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
        lap_u = (np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)) / (
            dx**2
        ) + (np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)) / (dy**2)
        pressure_grad_x = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2.0 * dx)
        u = u - dt * (u * dudx + v * dudy) - dt * pressure_grad_x + dt * nu * lap_u

        dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
        dvdy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dy)
        lap_v = (np.roll(v, -1, axis=0) - 2.0 * v + np.roll(v, 1, axis=0)) / (
            dx**2
        ) + (np.roll(v, -1, axis=1) - 2.0 * v + np.roll(v, 1, axis=1)) / (dy**2)
        pressure_grad_y = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * dy)
        v = v - dt * (u * dvdx + v * dvdy) - dt * pressure_grad_y + dt * nu * lap_v

        solid = geometry_mask == 1
        u[solid] = 0.0
        v[solid] = 0.0

        p = p + 0.05 * (u**2 + v**2)

        # Keep the explicit scheme numerically bounded for fast prototyping.
        u = np.nan_to_num(u, nan=0.0, posinf=10.0, neginf=-10.0)
        v = np.nan_to_num(v, nan=0.0, posinf=10.0, neginf=-10.0)
        p = np.nan_to_num(p, nan=0.0, posinf=50.0, neginf=-50.0)
        u = np.clip(u, -10.0, 10.0)
        v = np.clip(v, -10.0, 10.0)
        p = np.clip(p, -50.0, 50.0)

    vorticity = compute_vorticity(u=u, v=v, dx=dx, dy=dy)
    return {
        "velocity_field": {"u": u, "v": v},
        "pressure_field": p,
        "vorticity_field": vorticity,
        "geometry_mask": geometry_mask,
        "solver": "simplified_fd",
    }
