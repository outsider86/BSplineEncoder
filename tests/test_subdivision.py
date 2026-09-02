import numpy as np
from scipy.interpolate import make_interp_spline

from spline_encoder import (
    create_encoder,
    preset,
    subdivide,
    subdivide_parameters,
)


def test_exact_low_level_subdivision():
    x = np.arange(10, dtype=np.float64)
    y = np.stack((np.sin(x), np.cos(x)), axis=1)
    spline = make_interp_spline(x, y, k=3)
    left, right = subdivide(spline, 4.25)
    left_times = np.linspace(0, 4.25, 100)
    right_times = np.linspace(4.25, 9, 100)
    np.testing.assert_allclose(left(left_times), spline(left_times), atol=1e-12, rtol=0)
    np.testing.assert_allclose(right(right_times), spline(right_times), atol=1e-12, rtol=0)


def test_high_level_subdivision_preserves_uniform_curve():
    t = np.linspace(0, 5, 50)
    actions = np.stack([np.sin(t + i) for i in range(7)], axis=1)
    result = create_encoder(preset("piper_chunk50", "uniform_double")).encode_chunk(actions)
    left, right = subdivide_parameters(result, 25)
    original = result.decode()
    np.testing.assert_allclose(left.decode(), original[:25], atol=1e-12, rtol=0)
    np.testing.assert_allclose(right.decode(), original[25:], atol=1e-12, rtol=0)
