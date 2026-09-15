import numpy as np
import pytest
from scipy.interpolate import BSpline, make_interp_spline

from spline_encoder import (
    UniformDoubleBSplineConfig,
    create_encoder,
    extract_interval,
    subdivide,
)
from spline_encoder.encoder import _design
from spline_encoder.quantization import ControlQuantizer


@pytest.mark.parametrize("degree", range(1, 6))
def test_design_matrix_matches_scipy_and_partitions_unity(degree):
    num_basis = degree + 6
    breaks = np.linspace(0.0, 9.0, num_basis - degree + 1)
    knots = np.r_[np.repeat(breaks[0], degree), breaks, np.repeat(breaks[-1], degree)]
    samples = np.r_[np.linspace(0.0, 9.0, 91), breaks]
    actual = _design(samples, knots, degree)
    expected = BSpline.design_matrix(samples, knots, degree).toarray()
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, atol=2e-15, rtol=0)
    assert np.min(actual) >= 0.0


@pytest.mark.parametrize("degree", (1, 2, 3, 4, 5))
def test_unregularized_uniform_fit_recovers_a_known_spline(degree):
    rng = np.random.default_rng(1000 + degree)
    config = UniformDoubleBSplineConfig(
        action_dim=3,
        chunk_size=18,
        frequency_hz=25,
        degree=degree,
        num_basis=degree + 6,
        span_length_steps=None,
        regularization=0.0,
    )
    encoder = create_encoder(config)
    expected_controls = rng.normal(size=(config.num_basis, config.action_dim))
    times = np.arange(config.chunk_size, dtype=np.float64)
    actions = BSpline(encoder.knots, expected_controls, degree)(times)
    result = encoder.encode_chunk(actions)
    np.testing.assert_allclose(result.control_points, expected_controls, atol=2e-12, rtol=0)
    np.testing.assert_allclose(result.decode(), actions, atol=2e-12, rtol=0)


@pytest.mark.parametrize("degree", range(1, 6))
def test_random_subdivision_and_interval_extraction_preserve_curve(degree):
    rng = np.random.default_rng(2000 + degree)
    count = degree + 9
    times = np.arange(count, dtype=np.float64)
    spline = make_interp_spline(times, rng.normal(size=(count, 3)), k=degree)
    split = 2.375
    left, right = subdivide(spline, split)
    left_times = np.linspace(times[0], split, 137)
    right_times = np.linspace(split, times[-1], 137)
    np.testing.assert_allclose(left(left_times), spline(left_times), atol=4e-12, rtol=0)
    np.testing.assert_allclose(right(right_times), spline(right_times), atol=4e-12, rtol=0)

    interval_left, interval_right = 1.25, times[-1] - 1.125
    interval = extract_interval(spline, interval_left, interval_right)
    local_times = np.linspace(0.0, interval_right - interval_left, 151)
    np.testing.assert_allclose(
        interval(local_times), spline(local_times + interval_left), atol=6e-12, rtol=0
    )


def test_quantization_error_bound_clipping_and_constant_channel():
    quantizer = ControlQuantizer(17)
    low = np.asarray([-2.0, 4.0, 7.0])
    high = np.asarray([2.0, 12.0, 7.0])
    quantizer.set_bounds(low, high)
    values = np.asarray([
        [-3.0, 4.0, 7.0],
        [-0.75, 9.25, 7.0],
        [3.0, 13.0, 7.0],
    ])
    tokens = quantizer.encode(values)
    reconstructed = quantizer.decode(tokens)
    assert np.all((tokens >= 0) & (tokens < quantizer.vocab_size))
    np.testing.assert_array_equal(reconstructed[:, 2], 7.0)
    np.testing.assert_array_equal(reconstructed[[0, 2], 0], [-2.0, 2.0])
    np.testing.assert_array_equal(reconstructed[[0, 2], 1], [4.0, 12.0])
    in_range = np.asarray([[-0.75, 9.25, 7.0]])
    error = np.abs(quantizer.decode(quantizer.encode(in_range)) - in_range)
    half_bin = (high - low) / (2 * (quantizer.vocab_size - 1))
    assert np.all(error <= half_bin + 1e-15)
