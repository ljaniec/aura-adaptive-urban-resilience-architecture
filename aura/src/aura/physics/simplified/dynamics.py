"""Simplified finite-difference updates used by Level-1 simulations."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def laplacian(field: Array, dx: float, dy: float) -> Array:
    """Return a second-order central-difference Laplacian."""
    return (np.roll(field, -1, axis=0) - 2.0 * field + np.roll(field, 1, axis=0)) / (
        dx**2
    ) + (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / (
        dy**2
    )


def advect_scalar(
    scalar: Array, u: Array, v: Array, dt: float, dx: float, dy: float
) -> Array:
    """First-order explicit advection for a scalar field."""
    dsdx = (np.roll(scalar, -1, axis=0) - np.roll(scalar, 1, axis=0)) / (2.0 * dx)
    dsdy = (np.roll(scalar, -1, axis=1) - np.roll(scalar, 1, axis=1)) / (2.0 * dy)
    return scalar - dt * (u * dsdx + v * dsdy)


def pressure_smoothing(pressure: Array, iterations: int = 2) -> Array:
    """Diffuse pressure to emulate fast pressure propagation."""
    p = pressure.copy()
    for _ in range(iterations):
        p = 0.25 * (
            np.roll(p, 1, axis=0)
            + np.roll(p, -1, axis=0)
            + np.roll(p, 1, axis=1)
            + np.roll(p, -1, axis=1)
        )
    return p


def compute_vorticity(u: Array, v: Array, dx: float, dy: float) -> Array:
    """Compute 2D scalar vorticity from velocity components."""
    dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dx)
    dudy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
    return dvdx - dudy
