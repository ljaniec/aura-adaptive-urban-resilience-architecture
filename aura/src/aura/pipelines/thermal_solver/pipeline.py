from kedro.pipeline import Node, Pipeline

from .nodes import run_thermal_solver


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=run_thermal_solver,
                inputs=[
                    "indoor_flow_state",
                    "facade_state",
                    "params:simulation",
                    "params:domain",
                    "params:thermal",
                    "params:building",
                ],
                outputs="thermal_state",
                name="run_thermal_solver_node",
            )
        ]
    )
