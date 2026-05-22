"""Nodes for optimization loops over facade control variables."""

from __future__ import annotations

from aura.optimization.solvers import random_search_step


def run_optimization_loop(
    simulation_metrics: dict,
    facade_state: dict,
    optimization: dict,
    facade: dict,
) -> dict:
    """Run a light baseline optimization process with random-search perturbations."""
    iters = int(optimization["iterations"])
    seed = int(optimization.get("seed", 7))

    best = {
        "objective": float(simulation_metrics["objective"]),
        "panel_angle": list(facade_state["panel_angle"]),
        "panel_opening": list(facade_state["panel_opening"]),
        "iteration": 0,
    }
    history = [best.copy()]

    current = {
        "panel_angle": best["panel_angle"],
        "panel_opening": best["panel_opening"],
    }
    for i in range(1, iters + 1):
        candidate = random_search_step(current, facade_cfg=facade, seed=seed + i)
        angle_penalty = sum(abs(x) for x in candidate["panel_angle"]) / (
            len(candidate["panel_angle"]) + 1e-9
        )
        opening_penalty = sum(candidate["panel_opening"]) / (
            len(candidate["panel_opening"]) + 1e-9
        )
        candidate_objective = float(
            simulation_metrics["objective"]
            + 0.01 * angle_penalty
            + 0.05 * opening_penalty
        )

        entry = {
            "objective": candidate_objective,
            "panel_angle": candidate["panel_angle"],
            "panel_opening": candidate["panel_opening"],
            "iteration": i,
        }
        history.append(entry)

        if candidate_objective < best["objective"]:
            best = entry
            current = candidate

    return {
        "algorithm": optimization["algorithm"],
        "best_solution": best,
        "history": history,
    }
