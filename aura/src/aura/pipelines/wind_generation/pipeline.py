from kedro.pipeline import Node, Pipeline

from .nodes import generate_wind_field


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=generate_wind_field,
                inputs=["params:wind", "params:simulation", "params:domain"],
                outputs="wind_inlet",
                name="generate_wind_field_node",
            )
        ]
    )
