"""Nodes for synthetic wind boundary generation."""

from __future__ import annotations

import numpy as np


def generate_wind_field(wind: dict, simulation: dict, domain: dict) -> dict:
    """Generate inlet velocity profile u_inlet[t, y], v_inlet[t, y]."""
    steps = int(simulation["steps"])
    ny = int(domain["ny"])

    velocity_mean = float(wind["velocity_mean"])
    gust_strength = float(wind["gust_strength"])
    direction_deg = float(wind["direction_deg"])
    turbulence_intensity = float(wind["turbulence_intensity"])
    mode = wind.get("mode", "sinusoidal")

    t = np.arange(steps)
    if mode == "constant":
        base = np.full(steps, velocity_mean)
    elif mode == "stochastic":
        rng = np.random.default_rng(int(wind.get("seed", 123)))
        base = velocity_mean + rng.normal(scale=gust_strength, size=steps)
    elif mode == "urban_canyon_profile":
        omega = 2.0 * np.pi / max(steps, 1)
        base = velocity_mean + 0.6 * gust_strength * np.sin(omega * t)
    else:
        omega = 2.0 * np.pi / max(steps, 1)
        base = velocity_mean + gust_strength * np.sin(omega * t)

    if mode == "urban_canyon_profile":
        # Stronger shear at the top layer and mild near-ground slowdown.
        y = np.linspace(0.0, 1.0, ny)
        y_profile = 0.65 + 0.95 * (y**0.7)
        u_inlet = np.outer(base, y_profile)
    else:
        y_profile = np.linspace(0.8, 1.2, ny)
        u_inlet = np.outer(base, y_profile)

    rng = np.random.default_rng(int(wind.get("seed", 123)))
    v_inlet = turbulence_intensity * velocity_mean * rng.normal(size=(steps, ny))
    if mode == "urban_canyon_profile":
        y = np.linspace(0.0, 1.0, ny)
        # Negative vertical bias in upper band to seed downwash-like behavior.
        downwash = -0.08 * velocity_mean * np.clip(y - 0.55, 0.0, 1.0)
        v_inlet += downwash[None, :]

    theta = np.deg2rad(direction_deg)
    return {
        "u_inlet": (u_inlet * np.cos(theta)).astype(float),
        "v_inlet": (v_inlet + u_inlet * np.sin(theta)).astype(float),
        "mode": mode,
    }
