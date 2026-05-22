"""Geometry mask generation utilities.

Mask values:
0 = fluid
1 = solid wall/building
2 = facade panel
3 = inlet
4 = outlet
5 = ground
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def build_geometry_mask(
    nx: int,
    ny: int,
    building: dict,
    facade: dict,
) -> Array:
    """Create a coarse 2D geometry mask with building and facade strips."""
    mask = np.zeros((nx, ny), dtype=np.int8)

    b_width = int(building["width_cells"])
    b_height = int(building["height_cells"])
    ground_cells = max(1, int(building.get("ground_cells", 2)))
    on_ground = bool(building.get("on_ground", True))

    x_center = int(float(building.get("x_center_ratio", 0.5)) * nx)
    y_center = int(float(building.get("y_center_ratio", 0.5)) * ny)

    x0 = max(2, x_center - b_width // 2)
    x1 = min(nx - 2, x0 + b_width)
    if on_ground:
        y0 = ground_cells
    else:
        y0 = max(2, y_center - b_height // 2)
    y1 = min(ny - 2, y0 + b_height)

    # Ground strip at bottom for side-view urban street section.
    mask[:, :ground_cells] = 5

    mask[x0:x1, y0:y1] = 1

    panels = max(1, int(facade["num_panels"]))
    panel_stride = max(1, (y1 - y0) // panels)
    facade_x = x0 - 1

    for i in range(panels):
        py0 = y0 + i * panel_stride
        py1 = y1 if i == panels - 1 else min(y1, py0 + panel_stride)
        mask[facade_x, py0:py1] = 2

    mask[0, :] = 3
    mask[-1, :] = 4
    mask[:, :ground_cells] = 5
    return mask
