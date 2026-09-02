"""Observation-aligned continuous LeRobot dataset encoding example."""

from spline_encoder import (
    ActionAlignment,
    create_encoder,
    encode_lerobot_dataset,
    preset,
)


encoder = create_encoder(preset("piper_chunk50", "uniform_left"))
alignment = ActionAlignment(action_indices=(0, 1, 2, 3, 4, 5, 6))
result = encode_lerobot_dataset(
    "/path/to/lerobot/dataset",
    "encoded_splines/piper",
    encoder,
    alignment=alignment,
)
print(result)
