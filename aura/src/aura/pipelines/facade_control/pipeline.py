from kedro.pipeline import Node, Pipeline

from .nodes import compute_facade_state


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=compute_facade_state,
                inputs=[
                    "external_flow_state",
                    "params:facade",
                    "params:controllers",
                    "params:thermal",
                ],
                outputs="facade_state",
                name="compute_facade_state_node",
            )
        ]
    )
