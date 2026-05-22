"""Level-1 simplified fluid and thermal dynamics."""

from .dynamics import advect_scalar, compute_vorticity, laplacian, pressure_smoothing

__all__ = ["laplacian", "advect_scalar", "pressure_smoothing", "compute_vorticity"]
