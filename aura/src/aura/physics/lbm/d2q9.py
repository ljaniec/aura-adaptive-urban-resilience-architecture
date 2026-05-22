"""D2Q9 LBM solver for 2D external airflow.

This implementation is NumPy-based and intentionally lightweight so it can run
inside Kedro experiments without compiled dependencies.
"""

from __future__ import annotations

import numpy as np

from aura.physics.simplified import compute_vorticity

Array = np.ndarray

# D2Q9 discrete velocities and weights.
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=int,
)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
CS2 = 1.0 / 3.0


def equilibrium(rho: Array, ux: Array, uy: Array) -> Array:
    """Compute D2Q9 equilibrium distributions."""
    feq = np.zeros((9, *rho.shape), dtype=float)
    u2 = ux**2 + uy**2
    for i in range(9):
        cu = C[i, 0] * ux + C[i, 1] * uy
        feq[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2)
    return feq


def _apply_inlet_zou_he(f: Array, u_profile: Array, v_profile: Array) -> None:
    """Apply left-boundary velocity inlet with Zou/He reconstruction."""
    # Boundary column uses x=0 distributions, and interior x=1 for reference.
    rho_b = (
        f[0, 0, :]
        + f[2, 0, :]
        + f[4, 0, :]
        + 2.0 * (f[3, 0, :] + f[6, 0, :] + f[7, 0, :])
    ) / np.clip(1.0 - u_profile, 1e-6, None)

    f[1, 0, :] = f[3, 0, :] + (2.0 / 3.0) * rho_b * u_profile
    f[5, 0, :] = (
        f[7, 0, :]
        + 0.5 * (f[4, 0, :] - f[2, 0, :])
        + (1.0 / 6.0) * rho_b * u_profile
        + 0.5 * rho_b * v_profile
    )
    f[8, 0, :] = (
        f[6, 0, :]
        + 0.5 * (f[2, 0, :] - f[4, 0, :])
        + (1.0 / 6.0) * rho_b * u_profile
        - 0.5 * rho_b * v_profile
    )


def _apply_outlet_copy(f: Array) -> None:
    """Simple convective-like outlet by copying interior populations."""
    f[:, -1, :] = f[:, -2, :]


def run_lbm_d2q9(
    u_inlet: Array,
    v_inlet: Array,
    geometry_mask: Array,
    lbm_cfg: dict,
    dx: float,
    dy: float,
) -> dict:
    """Run a D2Q9 BGK simulation and recover flow/pressure/vorticity fields."""
    steps_requested = int(lbm_cfg.get("steps", u_inlet.shape[0]))
    min_steps = int(lbm_cfg.get("min_steps", steps_requested))
    steps = max(steps_requested, min_steps)
    tau = max(float(lbm_cfg.get("tau", 0.58)), 0.56)
    rho0 = float(lbm_cfg.get("rho0", 1.0))
    max_u = min(float(lbm_cfg.get("max_velocity", 0.18)), 0.18)
    panel_permeability = float(lbm_cfg.get("panel_permeability", 0.25))
    inlet_perturbation = float(lbm_cfg.get("inlet_perturbation", 0.0))
    perturbation_period = int(lbm_cfg.get("perturbation_period", 40))

    nx, ny = geometry_mask.shape
    u0 = np.clip(u_inlet[0, :], -max_u, max_u)
    v0 = np.clip(v_inlet[0, :], -max_u, max_u)
    ux = np.repeat(u0[None, :], nx, axis=0)
    uy = np.repeat(v0[None, :], nx, axis=0)
    rho = np.full((nx, ny), rho0, dtype=float)

    solid_wall = np.logical_or(geometry_mask == 1, geometry_mask == 5)
    panel_mask = geometry_mask == 2

    ux[solid_wall] = 0.0
    uy[solid_wall] = 0.0
    f = equilibrium(rho=rho, ux=ux, uy=uy)

    for t in range(steps):
        rho = np.sum(f, axis=0)
        rho = np.clip(np.nan_to_num(rho, nan=rho0), 0.2 * rho0, 5.0 * rho0)
        ux = np.sum(f * C[:, 0][:, None, None], axis=0) / rho
        uy = np.sum(f * C[:, 1][:, None, None], axis=0) / rho

        ux = np.clip(ux, -max_u, max_u)
        uy = np.clip(uy, -max_u, max_u)
        ux[solid_wall] = 0.0
        uy[solid_wall] = 0.0

        feq = equilibrium(rho=rho, ux=ux, uy=uy)
        f = f - (1.0 / tau) * (f - feq)

        # Bounce-back on solid walls.
        for i in range(9):
            bounced = f[OPP[i]]
            f[i, solid_wall] = bounced[solid_wall]

        # Partial bounce-back for facade panels to mimic controllable permeability.
        for i in range(9):
            bounced = f[OPP[i]]
            f[i, panel_mask] = (1.0 - panel_permeability) * bounced[
                panel_mask
            ] + panel_permeability * f[i, panel_mask]

        # Streaming step.
        for i in range(9):
            f[i] = np.roll(f[i], shift=C[i, 0], axis=0)
            f[i] = np.roll(f[i], shift=C[i, 1], axis=1)

        inlet_idx = min(t, u_inlet.shape[0] - 1)
        v_profile = np.clip(v_inlet[inlet_idx, :], -max_u, max_u)
        if inlet_perturbation > 0.0:
            # Mild periodic perturbation to break symmetry and reveal wake dynamics.
            y = np.linspace(0.0, 1.0, ny)
            shape = np.sin(np.pi * y)
            osc = np.sin(2.0 * np.pi * t / max(perturbation_period, 1))
            v_profile = np.clip(
                v_profile + inlet_perturbation * max_u * osc * shape,
                -max_u,
                max_u,
            )
        _apply_inlet_zou_he(
            f=f,
            u_profile=np.clip(u_inlet[inlet_idx, :], -max_u, max_u),
            v_profile=v_profile,
        )
        _apply_outlet_copy(f)

        # Open-top style boundary: weak copy at upper edge only.
        f[:, :, -1] = f[:, :, -2]

        # Keep particle distributions physically bounded.
        f = np.nan_to_num(f, nan=0.0, posinf=1e3, neginf=0.0)
        f = np.clip(f, 0.0, 1e3)

    rho = np.sum(f, axis=0)
    rho = np.clip(np.nan_to_num(rho, nan=rho0), 0.2 * rho0, 5.0 * rho0)
    ux = np.sum(f * C[:, 0][:, None, None], axis=0) / rho
    uy = np.sum(f * C[:, 1][:, None, None], axis=0) / rho
    ux = np.clip(np.nan_to_num(ux, nan=0.0), -max_u, max_u)
    uy = np.clip(np.nan_to_num(uy, nan=0.0), -max_u, max_u)
    ux[solid_wall] = 0.0
    uy[solid_wall] = 0.0

    pressure = CS2 * rho
    vorticity = compute_vorticity(ux, uy, dx=dx, dy=dy)

    return {
        "velocity_field": {"u": ux, "v": uy},
        "pressure_field": pressure,
        "vorticity_field": vorticity,
        "geometry_mask": geometry_mask,
        "solver": "lbm_d2q9",
    }
