from kedro.pipeline import Node, Pipeline

from .nodes import run_internal_cfd


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=run_internal_cfd,
                inputs=[
                    "external_flow_state",
                    "facade_state",
                    "params:simulation",
                    "params:domain",
                    "params:fluid",
                    "params:building",
                    "params:facade",
                ],
                outputs="indoor_flow_state",
                name="run_internal_cfd_node",
            )
        ]
    )
