from kedro.pipeline import Node, Pipeline

from .nodes import compute_simulation_metrics


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=compute_simulation_metrics,
                inputs=[
                    "external_flow_state",
                    "indoor_flow_state",
                    "thermal_state",
                    "facade_state",
                    "params:metrics",
                ],
                outputs="simulation_metrics",
                name="compute_simulation_metrics_node",
            )
        ]
    )
