from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_simulation_summary,
    create_dashboard_figure,
    create_output_visualizations,
    create_step1_effects_figure,
    create_step2_thermal_figure,
    create_step3_facade_figure,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=create_output_visualizations,
                inputs=[
                    "wind_inlet",
                    "external_flow_state",
                    "facade_state",
                    "indoor_flow_state",
                    "thermal_state",
                    "simulation_metrics",
                    "optimization_result",
                ],
                outputs=[
                    "viz_wind_inlet",
                    "viz_external_flow",
                    "viz_facade_state",
                    "viz_indoor_flow",
                    "viz_thermal_state",
                    "viz_metrics",
                    "viz_optimization",
                ],
                name="create_output_visualizations_node",
            ),
            Node(
                func=create_dashboard_figure,
                inputs=[
                    "wind_inlet",
                    "external_flow_state",
                    "facade_state",
                    "indoor_flow_state",
                    "thermal_state",
                    "simulation_metrics",
                    "optimization_result",
                ],
                outputs="viz_dashboard",
                name="create_dashboard_figure_node",
            ),
            Node(
                func=create_step1_effects_figure,
                inputs=["wind_inlet", "external_flow_state"],
                outputs="viz_step1_effects",
                name="create_step1_effects_figure_node",
            ),
            Node(
                func=create_step2_thermal_figure,
                inputs=["external_flow_state", "indoor_flow_state", "thermal_state"],
                outputs="viz_step2_thermal",
                name="create_step2_thermal_figure_node",
            ),
            Node(
                func=create_step3_facade_figure,
                inputs=["external_flow_state", "facade_state", "thermal_state"],
                outputs="viz_step3_facade",
                name="create_step3_facade_figure_node",
            ),
            Node(
                func=build_simulation_summary,
                inputs=["simulation_metrics", "optimization_result", "facade_state"],
                outputs="simulation_summary",
                name="build_simulation_summary_node",
            ),
        ]
    )
