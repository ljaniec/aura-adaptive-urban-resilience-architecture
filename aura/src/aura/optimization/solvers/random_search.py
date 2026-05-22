"""Random-search baseline for optimization pipeline bootstrapping."""

from __future__ import annotations

import random


def random_search_step(base_state: dict, facade_cfg: dict, seed: int) -> dict:
    """Perturb panel angles/openings within valid ranges."""
    rng = random.Random(seed)

    angle_min = float(facade_cfg["angle_min"])
    angle_max = float(facade_cfg["angle_max"])
    opening_min = float(facade_cfg["opening_min"])
    opening_max = float(facade_cfg["opening_max"])

    angles = [
        max(angle_min, min(angle_max, a + rng.uniform(-5.0, 5.0)))
        for a in base_state["panel_angle"]
    ]
    openings = [
        max(opening_min, min(opening_max, o + rng.uniform(-0.08, 0.08)))
        for o in base_state["panel_opening"]
    ]

    return {"panel_angle": angles, "panel_opening": openings}
