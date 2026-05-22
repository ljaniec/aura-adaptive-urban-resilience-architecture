"""Nodes that format outputs for reporting and plotting layers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def create_output_visualizations(
    wind_inlet: dict,
    external_flow_state: dict,
    facade_state: dict,
    indoor_flow_state: dict,
    thermal_state: dict,
    simulation_metrics: dict,
    optimization_result: dict,
):
    """Create one visualization artifact for each pipeline output."""
    wind_fig = _plot_wind_inlet(wind_inlet)
    external_fig = _plot_external_flow(external_flow_state)
    facade_fig = _plot_facade_state(facade_state)
    indoor_fig = _plot_indoor_flow(indoor_flow_state)
    thermal_fig = _plot_thermal_state(thermal_state)
    metrics_fig = _plot_metrics(simulation_metrics)
    optimization_fig = _plot_optimization(optimization_result)
    return (
        wind_fig,
        external_fig,
        facade_fig,
        indoor_fig,
        thermal_fig,
        metrics_fig,
        optimization_fig,
    )


def create_step1_effects_figure(wind_inlet: dict, external_flow_state: dict):
    """Create PoC figure for Step 1: rectangle building and tunnel-like effects."""
    u = np.asarray(external_flow_state["velocity_field"]["u"])
    v = np.asarray(external_flow_state["velocity_field"]["v"])
    vort = np.asarray(external_flow_state["vorticity_field"])
    mask = np.asarray(external_flow_state["geometry_mask"])
    speed = np.sqrt(u**2 + v**2)

    fig, axes = plt.subplots(1, 3, figsize=(28, 10), constrained_layout=True)
    fig.patch.set_facecolor("#0B1220")
    ax0, ax1, ax2 = axes
    nx, ny = u.shape

    ax0.set_facecolor("#07101C")
    x = np.arange(u.shape[0])
    y = np.arange(u.shape[1])
    stream = ax0.streamplot(
        x,
        y,
        u.T,
        v.T,
        density=1.6,
        color=speed.T,
        cmap="RdBu_r",
        linewidth=1.3,
        arrowsize=0.8,
        minlength=0.2,
    )
    _overlay_solids(ax0, mask)
    ax0.set_title("Urban Wind Streamlines", color="white", pad=12, fontsize=17)
    ax0.set_xlabel("x [cells]", color="white", fontsize=13)
    ax0.set_ylabel("y [cells]", color="white", fontsize=13)
    ax0.tick_params(colors="white", labelsize=11)
    _set_step1_axes_geometry(ax0, nx=nx, ny=ny)
    _style_axes(ax0)
    c0 = fig.colorbar(stream.lines, ax=ax0, fraction=0.012, pad=0.01, shrink=0.58)
    c0.set_label("speed [lattice units]", color="white", fontsize=9)
    c0.ax.tick_params(colors="white", labelsize=8)

    ax1.set_facecolor("#07101C")
    fluid_mask = np.logical_not(np.logical_or(mask == 1, mask == 5))
    vort_abs = np.abs(vort[fluid_mask])
    vlim = max(float(np.percentile(vort_abs, 99.0)), 1e-4)
    im1 = ax1.imshow(
        vort.T,
        origin="lower",
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vlim,
        vmax=vlim,
    )
    _overlay_solids(ax1, mask)
    ax1.set_title("Vorticity and Wake Structure", color="white", pad=10, fontsize=15)
    ax1.set_xlabel("x [cells]", color="white", fontsize=12)
    ax1.set_ylabel("y [cells]", color="white", fontsize=12)
    ax1.tick_params(colors="white", labelsize=10)
    _set_step1_axes_geometry(ax1, nx=nx, ny=ny)
    _style_axes(ax1)
    c1 = fig.colorbar(im1, ax=ax1, fraction=0.018, pad=0.015, shrink=0.58)
    c1.set_label("vorticity [1/step]", color="white", fontsize=9)
    c1.ax.tick_params(colors="white", labelsize=8)

    ax2.set_facecolor("#07101C")
    inlet_u_ref = max(
        float(np.mean(np.clip(wind_inlet["u_inlet"][0, :], 1e-6, None))), 1e-6
    )
    speed_up = np.clip(speed / inlet_u_ref - 1.0, -1.0, 2.0)
    downwash = np.clip(-v / inlet_u_ref, 0.0, 1.5)
    effects_map = speed_up + 0.75 * downwash
    im2 = ax2.imshow(effects_map.T, origin="lower", cmap="magma", aspect="auto")
    _overlay_solids(ax2, mask)
    ax2.set_title("Tunnel Effects Intensity", color="white", pad=10, fontsize=15)
    ax2.set_xlabel("x [cells]", color="white", fontsize=12)
    ax2.set_ylabel("y [cells]", color="white", fontsize=12)
    ax2.tick_params(colors="white", labelsize=10)
    _set_step1_axes_geometry(ax2, nx=nx, ny=ny)
    _style_axes(ax2)
    c2 = fig.colorbar(im2, ax=ax2, fraction=0.018, pad=0.015, shrink=0.58)
    c2.set_label("speed-up + downwash index [-]", color="white", fontsize=9)
    c2.ax.tick_params(colors="white", labelsize=8)

    bx0, bx1, by0, by1 = _building_bounds(mask)
    if bx0 is not None:
        _annotate_step1_effects(
            ax0, bx0=bx0, bx1=bx1, by0=by0, by1=by1, preset="stream"
        )
        _annotate_step1_effects(
            ax1, bx0=bx0, bx1=bx1, by0=by0, by1=by1, preset="vorticity"
        )
        _annotate_step1_effects(
            ax2, bx0=bx0, bx1=bx1, by0=by0, by1=by1, preset="tunnel"
        )

    mode = wind_inlet.get("mode", "unknown")
    fig.suptitle(
        f"Urban Wind PoC - Step 1 (Rectangular Building, wind mode: {mode})",
        fontsize=21,
        color="white",
    )
    return fig


def create_step2_thermal_figure(
    external_flow_state: dict,
    indoor_flow_state: dict,
    thermal_state: dict,
):
    """Create PoC figure for Step 2: airflow + thermal transport."""
    u_ext = np.asarray(external_flow_state["velocity_field"]["u"])
    v_ext = np.asarray(external_flow_state["velocity_field"]["v"])
    u_in = np.asarray(indoor_flow_state["indoor_velocity"]["u"])
    v_in = np.asarray(indoor_flow_state["indoor_velocity"]["v"])
    t_field = np.asarray(thermal_state["temperature_field"])
    mask = np.asarray(external_flow_state["geometry_mask"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    ext_speed = np.sqrt(u_ext**2 + v_ext**2)
    in_speed = np.sqrt(u_in**2 + v_in**2)

    im0 = axes[0].imshow(ext_speed.T, origin="lower", cmap="magma", aspect="auto")
    _overlay_solids(axes[0], mask)
    axes[0].set_title("Step 2: External Speed")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(in_speed.T, origin="lower", cmap="viridis", aspect="auto")
    _overlay_solids(axes[1], mask)
    axes[1].set_title("Step 2: Indoor Speed")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(t_field.T, origin="lower", cmap="inferno", aspect="auto")
    _overlay_solids(axes[2], mask)
    axes[2].set_title("Step 2: Temperature Field")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Urban Wind PoC - Step 2 (Thermal Coupling)", fontsize=14)
    return fig


def create_step3_facade_figure(
    external_flow_state: dict,
    facade_state: dict,
    thermal_state: dict,
):
    """Create PoC figure for Step 3: static facade strategy and outcomes."""
    p = np.asarray(external_flow_state["pressure_field"])
    openings = np.asarray(facade_state["panel_opening"])
    angles = np.asarray(facade_state["panel_angle"])
    t_field = np.asarray(thermal_state["temperature_field"])
    mask = np.asarray(external_flow_state["geometry_mask"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axes[0].imshow(p.T, origin="lower", cmap="coolwarm", aspect="auto")
    _overlay_solids(axes[0], mask)
    axes[0].set_title("Step 3: Pressure Field")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    idx = np.arange(len(openings))
    axes[1].bar(idx, openings, label="opening", color="#EF6C00", alpha=0.85)
    axes[1].plot(idx, angles, label="angle", color="#6A1B9A")
    axes[1].set_title("Step 3: Static Facade State")
    axes[1].set_xlabel("panel")
    axes[1].legend(loc="best")

    im2 = axes[2].imshow(t_field.T, origin="lower", cmap="inferno", aspect="auto")
    _overlay_solids(axes[2], mask)
    axes[2].set_title("Step 3: Thermal Outcome")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Urban Wind PoC - Step 3 (Static Facade)", fontsize=14)
    return fig


def create_dashboard_figure(
    wind_inlet: dict,
    external_flow_state: dict,
    facade_state: dict,
    indoor_flow_state: dict,
    thermal_state: dict,
    simulation_metrics: dict,
    optimization_result: dict,
):
    """Create a single dashboard-style figure combining all pipeline outputs."""
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)

    ax_wind = fig.add_subplot(grid[0, 0])
    u = np.asarray(wind_inlet["u_inlet"])
    v = np.asarray(wind_inlet["v_inlet"])
    ax_wind.plot(np.mean(u, axis=1), color="#1565C0", label="u")
    ax_wind.plot(np.mean(v, axis=1), color="#2E7D32", label="v")
    ax_wind.set_title("Wind")
    ax_wind.set_xlabel("t")
    ax_wind.legend(loc="best")

    ax_ext = fig.add_subplot(grid[0, 1])
    p = np.asarray(external_flow_state["pressure_field"])
    im_ext = ax_ext.imshow(p.T, origin="lower", cmap="coolwarm", aspect="auto")
    ax_ext.set_title("External Pressure")
    fig.colorbar(im_ext, ax=ax_ext, fraction=0.046, pad=0.04)

    ax_facade = fig.add_subplot(grid[0, 2])
    angles = np.asarray(facade_state["panel_angle"])
    openings = np.asarray(facade_state["panel_opening"])
    idx = np.arange(len(angles))
    ax_facade.plot(idx, angles, color="#8E24AA", label="angle")
    ax_facade.plot(idx, openings, color="#EF6C00", label="opening")
    ax_facade.set_title("Facade State")
    ax_facade.set_xlabel("panel")
    ax_facade.legend(loc="best")

    ax_in = fig.add_subplot(grid[0, 3])
    iu = np.asarray(indoor_flow_state["indoor_velocity"]["u"])
    iv = np.asarray(indoor_flow_state["indoor_velocity"]["v"])
    speed = np.sqrt(iu**2 + iv**2)
    im_in = ax_in.imshow(speed.T, origin="lower", cmap="viridis", aspect="auto")
    ax_in.set_title("Indoor Speed")
    fig.colorbar(im_in, ax=ax_in, fraction=0.046, pad=0.04)

    ax_th = fig.add_subplot(grid[1, 0])
    temp = np.asarray(thermal_state["temperature_field"])
    im_th = ax_th.imshow(temp.T, origin="lower", cmap="inferno", aspect="auto")
    ax_th.set_title("Temperature")
    fig.colorbar(im_th, ax=ax_th, fraction=0.046, pad=0.04)

    ax_metrics = fig.add_subplot(grid[1, 1])
    metric_keys = [
        "mean_indoor_temp",
        "air_exchange_rate",
        "pressure_drop",
        "drag_force",
        "vortex_intensity",
        "objective",
    ]
    metric_values = [float(simulation_metrics[k]) for k in metric_keys]
    ax_metrics.bar(np.arange(len(metric_keys)), metric_values, color="#3949AB")
    ax_metrics.set_xticks(np.arange(len(metric_keys)))
    ax_metrics.set_xticklabels(metric_keys, rotation=30, ha="right")
    ax_metrics.set_title("Metrics")

    ax_opt = fig.add_subplot(grid[1, 2])
    history = optimization_result["history"]
    iters = [entry["iteration"] for entry in history]
    objs = [float(entry["objective"]) for entry in history]
    best_iter = optimization_result["best_solution"]["iteration"]
    ax_opt.plot(iters, objs, marker="o", color="#00897B")
    ax_opt.axvline(best_iter, color="#D81B60", linestyle="--")
    ax_opt.set_title("Optimization")
    ax_opt.set_xlabel("iter")

    ax_text = fig.add_subplot(grid[1, 3])
    ax_text.axis("off")
    ax_text.set_title("Summary")
    ax_text.text(
        0.0,
        0.95,
        "\n".join(
            [
                f"Algorithm: {optimization_result['algorithm']}",
                f"Best objective: {optimization_result['best_solution']['objective']:.4f}",
                f"Best iteration: {optimization_result['best_solution']['iteration']}",
                f"Mean indoor temp: {simulation_metrics['mean_indoor_temp']:.2f}",
                f"Air exchange rate: {simulation_metrics['air_exchange_rate']:.4f}",
            ]
        ),
        va="top",
    )

    fig.suptitle("Adaptive Facade Experiment Dashboard", fontsize=16)
    return fig


def build_simulation_summary(
    simulation_metrics: dict,
    optimization_result: dict,
    facade_state: dict,
) -> dict:
    """Generate a compact report artifact for dashboards/notebooks."""
    return {
        "metrics": simulation_metrics,
        "optimization": {
            "algorithm": optimization_result["algorithm"],
            "best_objective": optimization_result["best_solution"]["objective"],
            "best_iteration": optimization_result["best_solution"]["iteration"],
            "history_length": len(optimization_result["history"]),
        },
        "facade": {
            "mode": facade_state["mode"],
            "panel_count": len(facade_state["panel_angle"]),
            "angles": facade_state["panel_angle"],
            "openings": facade_state["panel_opening"],
        },
    }


def _plot_wind_inlet(wind_inlet: dict):
    u = np.asarray(wind_inlet["u_inlet"])
    v = np.asarray(wind_inlet["v_inlet"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(np.mean(u, axis=1), color="#1565C0")
    axes[0].set_title("Wind Generator Output: Mean Inlet u(t)")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("u")
    axes[1].plot(np.mean(v, axis=1), color="#2E7D32")
    axes[1].set_title("Wind Generator Output: Mean Inlet v(t)")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("v")
    return fig


def _plot_external_flow(external_flow_state: dict):
    p = np.asarray(external_flow_state["pressure_field"])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(p.T, origin="lower", cmap="coolwarm", aspect="auto")
    fig.colorbar(im, ax=ax, label="Pressure")
    ax.set_title("External CFD Output: Pressure Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def _plot_facade_state(facade_state: dict):
    angles = np.asarray(facade_state["panel_angle"])
    openings = np.asarray(facade_state["panel_opening"])
    idx = np.arange(len(angles))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].bar(idx, angles, color="#8E24AA")
    axes[0].set_title("Facade Controller Output: Panel Angles")
    axes[0].set_xlabel("Panel index")
    axes[0].set_ylabel("Angle (deg)")
    axes[1].bar(idx, openings, color="#EF6C00")
    axes[1].set_title("Facade Controller Output: Panel Openings")
    axes[1].set_xlabel("Panel index")
    axes[1].set_ylabel("Opening ratio")
    return fig


def _plot_indoor_flow(indoor_flow_state: dict):
    u = np.asarray(indoor_flow_state["indoor_velocity"]["u"])
    v = np.asarray(indoor_flow_state["indoor_velocity"]["v"])
    speed = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(speed.T, origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Speed")
    ax.set_title("Internal CFD Output: Indoor Speed Magnitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def _plot_thermal_state(thermal_state: dict):
    temperature = np.asarray(thermal_state["temperature_field"])
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    im = ax.imshow(temperature.T, origin="lower", cmap="inferno", aspect="auto")
    fig.colorbar(im, ax=ax, label="Temperature")
    ax.set_title("Thermal Solver Output: Temperature Field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def _plot_metrics(simulation_metrics: dict):
    keys = [
        "mean_indoor_temp",
        "air_exchange_rate",
        "pressure_drop",
        "drag_force",
        "vortex_intensity",
        "pedestrian_wind_comfort",
        "objective",
    ]
    values = [float(simulation_metrics[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.bar(keys, values, color="#3949AB")
    ax.set_title("Metrics Pipeline Output")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=35)
    return fig


def _plot_optimization(optimization_result: dict):
    history = optimization_result["history"]
    iters = [entry["iteration"] for entry in history]
    objs = [float(entry["objective"]) for entry in history]
    best_iter = optimization_result["best_solution"]["iteration"]

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(iters, objs, marker="o", color="#00897B")
    ax.axvline(best_iter, color="#D81B60", linestyle="--", label="best")
    ax.set_title("Optimization Pipeline Output: Objective History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.legend()
    return fig


def _overlay_solids(ax, mask: np.ndarray) -> None:
    solid_mask = np.logical_or(mask == 1, mask == 5)
    solid = np.ma.masked_where(~solid_mask, np.ones_like(mask, dtype=float))
    ax.imshow(solid.T, origin="lower", cmap="binary_r", alpha=0.95, aspect="auto")


def _set_step1_axes_geometry(ax, nx: int, ny: int) -> None:
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ny - 1)
    ax.set_box_aspect(ny / max(nx, 1))


def _annotate_step1_effects(
    ax,
    bx0: int,
    bx1: int,
    by0: int,
    by1: int,
    preset: str,
) -> None:
    cy = int(0.5 * (by0 + by1))
    ux = max(1, bx0 - 3)
    ay = by1 + 5

    placements = {
        "stream": {
            "down": (max(8, bx0 - 120), max(12, by0 + 10)),
            "corner": (bx0 + 24, by1 + 20),
            "venturi": (bx1 + 28, cy - 22),
        },
        "vorticity": {
            "down": (max(8, bx0 - 110), max(12, by0 + 8)),
            "corner": (bx0 + 20, by1 + 14),
            "venturi": (bx1 + 24, cy - 14),
        },
        "tunnel": {
            "down": (ux + 10, ay + 30),
            "corner": (bx0 + 16, by1 + 8),
            "venturi": (bx1 + 16, cy - 24),
        },
    }
    p = placements.get(preset, placements["stream"])
    ann_style = {
        "arrowprops": {"arrowstyle": "->", "color": "#F8FAFC", "lw": 1.4},
        "color": "#F8FAFC",
        "fontsize": 11,
        "bbox": {"facecolor": "#00000099", "edgecolor": "none", "pad": 2.0},
    }

    ax.annotate("Downdraught proxy", xy=(ux, ay), xytext=p["down"], **ann_style)
    ax.annotate("Corner acceleration", xy=(bx0, by1), xytext=p["corner"], **ann_style)
    ax.annotate(
        "Venturi-like speed-up",
        xy=(bx1 + 5, cy),
        xytext=p["venturi"],
        **ann_style,
    )


def _style_axes(ax) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")


def _building_bounds(mask: np.ndarray):
    yy, xx = np.where(mask == 1)
    if yy.size == 0:
        return None, None, None, None
    return int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max())
