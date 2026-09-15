import numpy as np
import pytest

from spline_encoder import (
    AdaptiveLeftBSplineConfig,
    SplineParameters,
    UniformDoubleBSplineConfig,
)
from spline_encoder.adaptive import encode_knot_durations, fit_adaptive_bspline
from spline_encoder.quantization import ControlQuantizer


@pytest.mark.parametrize(
    "field,value",
    (
        ("action_dim", 2.5),
        ("chunk_size", True),
        ("degree", 3.0),
        ("num_basis", 7.5),
        ("vocab_size", 256.0),
        ("frequency_hz", np.nan),
        ("frequency_hz", np.inf),
        ("regularization", np.nan),
    ),
)
def test_uniform_configuration_rejects_nonintegral_or_nonfinite_values(field, value):
    values = dict(
        action_dim=2,
        chunk_size=8,
        frequency_hz=10.0,
        degree=3,
        num_basis=7,
        span_length_steps=2,
        vocab_size=256,
        regularization=1e-4,
    )
    values[field] = value
    with pytest.raises(ValueError):
        UniformDoubleBSplineConfig(**values)


def test_quantizer_constructor_validates_and_copies_initial_bounds():
    low = np.asarray([0.0, 2.0])
    high = np.asarray([1.0, 4.0])
    quantizer = ControlQuantizer(8, low=low, high=high)
    low[:] = -100
    high[:] = 100
    np.testing.assert_array_equal(quantizer.low, [0.0, 2.0])
    np.testing.assert_array_equal(quantizer.high, [1.0, 4.0])
    with pytest.raises(ValueError, match="integer"):
        ControlQuantizer(1)
    with pytest.raises(ValueError, match="together"):
        ControlQuantizer(8, low=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        quantizer.encode([[np.nan, 2.0]])


def test_adaptive_helpers_reject_nonfinite_or_nonintegral_geometry_inputs():
    actions = np.stack((np.arange(8), np.arange(8) ** 2), axis=1)
    with pytest.raises(ValueError, match="timestamps"):
        fit_adaptive_bspline(actions, timestamps=[0, 1, 2, 3, np.nan, 5, 6, 7])
    with pytest.raises(ValueError, match="weights"):
        fit_adaptive_bspline(actions, weights=[1, 1, 1, 1, np.inf, 1, 1, 1])
    with pytest.raises(ValueError, match="FITPACK"):
        fit_adaptive_bspline(actions, degree=3.0)
    with pytest.raises(ValueError, match="integer"):
        fit_adaptive_bspline(actions, max_control_points=6.5)
    with pytest.raises(ValueError, match="contain integers"):
        fit_adaptive_bspline(actions, knot_insertion_excluded_dimensions=(1.5,))
    with pytest.raises(ValueError, match="invalid knots"):
        encode_knot_durations([0.0, np.nan], delta_t=0.1)
    with pytest.raises(ValueError, match="positive integer"):
        encode_knot_durations([0.0, 0.1], delta_t=0.1, max_duration=2.5)


def test_adaptive_configuration_does_not_truncate_excluded_dimensions():
    with pytest.raises(ValueError, match="contain integers"):
        AdaptiveLeftBSplineConfig(
            action_dim=2,
            chunk_size=8,
            degree=3,
            num_basis=7,
            span_length_steps=2,
            knot_insertion_excluded_dimensions=(1.5,),
        )


def test_spline_parameters_reject_invalid_domain_and_fractional_tokens():
    common = dict(
        control_points=np.zeros((2, 1)),
        degree=1,
        sample_period=0.1,
        executable_steps=2,
        tokenizer_id="test",
    )
    with pytest.raises(ValueError, match="positive length"):
        SplineParameters(knots=np.zeros(4), **common)
    with pytest.raises(ValueError, match="integer dtype"):
        SplineParameters(
            knots=np.asarray([0.0, 0.0, 1.0, 1.0]),
            tokens=np.zeros((2, 1), dtype=np.float64),
            **common,
        )
