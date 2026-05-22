from kedro.pipeline import Node, Pipeline

from .nodes import run_external_cfd


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=run_external_cfd,
                inputs=[
                    "wind_inlet",
                    "params:simulation",
                    "params:domain",
                    "params:fluid",
                    "params:building",
                    "params:facade",
                    "params:external_cfd",
                ],
                outputs="external_flow_state",
                name="run_external_cfd_node",
            )
        ]
    )
