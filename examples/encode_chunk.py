"""Minimal continuous Piper chunk encoding example."""

import numpy as np

from spline_encoder import create_encoder, preset


config = preset("piper_chunk50", "uniform_left")
encoder = create_encoder(config)
# Direct uniform-left consumes exactly chunk_size (50) actions; the legacy
# extended fit would report 56 here.
actions = np.zeros((config.input_chunk_size, config.action_dim), dtype=np.float64)
parameters = encoder.encode_chunk(actions)
print(parameters.control_points.shape, parameters.knots.shape, parameters.decode().shape)
