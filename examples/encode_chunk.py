"""Minimal continuous Piper chunk encoding example."""

import numpy as np

from spline_encoder import create_encoder, preset


actions = np.zeros((56, 7), dtype=np.float64)
encoder = create_encoder(preset("piper_chunk50", "uniform_left"))
parameters = encoder.encode_chunk(actions)
print(parameters.control_points.shape, parameters.knots.shape, parameters.decode().shape)
