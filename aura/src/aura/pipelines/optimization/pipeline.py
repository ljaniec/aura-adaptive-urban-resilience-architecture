from kedro.pipeline import Node, Pipeline

from .nodes import run_optimization_loop


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=run_optimization_loop,
                inputs=[
                    "simulation_metrics",
                    "facade_state",
                    "params:optimization",
                    "params:facade",
                ],
                outputs="optimization_result",
                name="run_optimization_loop_node",
            )
        ]
    )
