"""Project pipelines."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines(raise_errors=True)
    adaptive_names = [
        "wind_generation",
        "external_cfd",
        "facade_control",
        "internal_cfd",
        "thermal_solver",
        "metrics",
        "optimization",
        "visualization",
    ]
    pipelines["adaptive_facade_experiment"] = sum(
        (pipelines[name] for name in adaptive_names),
        Pipeline([]),
    )
    pipelines["step1_poc"] = (
        pipelines["wind_generation"]
        + pipelines["external_cfd"]
        + pipelines["visualization"].only_nodes("create_step1_effects_figure_node")
    )
    pipelines["step2_poc"] = (
        pipelines["wind_generation"]
        + pipelines["external_cfd"]
        + pipelines["facade_control"]
        + pipelines["internal_cfd"]
        + pipelines["thermal_solver"]
        + pipelines["visualization"].only_nodes("create_step2_thermal_figure_node")
    )
    pipelines["step3_poc"] = (
        pipelines["wind_generation"]
        + pipelines["external_cfd"]
        + pipelines["facade_control"]
        + pipelines["internal_cfd"]
        + pipelines["thermal_solver"]
        + pipelines["visualization"].only_nodes("create_step3_facade_figure_node")
    )
    pipelines["__default__"] = sum(  # pyright: ignore[reportArgumentType]
        pipelines.values()
    )  # pyright: ignore[reportArgumentType]
    return pipelines
