"""Left-clamped prefix extraction, suffix-to-prefix transfer, and requantization.

For degree ``p`` a left-clamped spline with knots ``[0]*(p+1) + u_1..u_k``
and ``k`` controls determines exactly its first ``k - p`` spans: span ``m``
uses controls ``m..m+p`` and distinct knots ``u_{m-p+1}..u_{m+p}``.  The last
duration ``u_k - u_{k-1}`` therefore only shapes right-side support.
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline

from spline_encoder import (
    AdaptiveLeftBSplineConfig,
    create_encoder,
    extract_left_clamped_prefix,
    preset,
)
from spline_encoder.subdivision import LeftClampedPrefix

DEGREES = range(1, 6)


def left_clamped(controls, durations, degree):
    distinct = np.r_[0.0, np.cumsum(durations)]
    knots = np.r_[np.zeros(degree + 1), distinct[1:]]
    return BSpline(knots, controls, degree, extrapolate=False, axis=0), distinct


def smooth_episode(length, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    return np.stack([
        np.sin(2 * np.pi * rng.uniform(0.5, 1.5) * (index + 1) * t + rng.uniform(0, 3))
        for index in range(dim)
    ], axis=1)


@pytest.mark.parametrize("degree", DEGREES)
def test_k_complete_pairs_preserve_exactly_the_first_k_minus_p_spans(degree):
    rng = np.random.default_rng(800 + degree)
    total = degree + 9
    controls = rng.normal(size=(total, 2))
    durations = rng.integers(1, 5, total).astype(np.float64)
    long_spline, distinct = left_clamped(controls, durations, degree)
    for k in range(degree + 1, total + 1):
        prefix = LeftClampedPrefix(distinct[: k + 1], controls[:k], degree)
        committed_end = distinct[k - degree]
        assert prefix.num_executable_spans == k - degree
        x = np.linspace(0.0, committed_end, 173)
        np.testing.assert_allclose(prefix.executable_spline()(x), long_spline(x), atol=1e-13, rtol=0)

        # Later controls, and every duration from the k-th on, cannot move
        # the committed interval.
        later_controls = controls.copy()
        later_controls[k:] += rng.normal(size=later_controls[k:].shape)
        later_durations = durations.copy()
        later_durations[k - 1:] += rng.integers(1, 4, total - k + 1)
        perturbed, _ = left_clamped(later_controls, later_durations, degree)
        np.testing.assert_allclose(perturbed(x), long_spline(x), atol=1e-13, rtol=0)

        # The bound is tight: control k-1 and duration k-1 both shape the
        # last committed span.
        last_span = np.linspace(distinct[k - degree - 1], committed_end, 31)[1:-1]
        moved_control = controls.copy()
        moved_control[k - 1] += 1.0
        assert np.abs(left_clamped(moved_control, durations, degree)[0](last_span)
                      - long_spline(last_span)).max() > 1e-6
        moved_duration = durations.copy()
        moved_duration[k - 2] += 1.0
        assert np.abs(left_clamped(controls, moved_duration, degree)[0](last_span)
                      - long_spline(last_span)).max() > 1e-6


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("cut", ("existing", "between", "start"))
def test_suffix_to_prefix_transfer_preserves_the_committed_interval(degree, cut):
    rng = np.random.default_rng(900 + degree)
    total = degree + 10
    controls = rng.normal(size=(total, 3))
    durations = rng.integers(1, 4, total).astype(np.float64)
    source, distinct = left_clamped(controls, durations, degree)
    executable_end = distinct[total - degree]
    tau = {"existing": distinct[3], "between": 0.5 * (distinct[3] + distinct[4]),
           "start": 0.0}[cut]
    available = len(distinct) - 1 - int(np.searchsorted(distinct, tau, side="right") - 1)
    prefix = extract_left_clamped_prefix(source, tau, num_knot_spans=available)
    transferred = prefix.executable_spline()

    # The transferred executable horizon ends where the source's did.
    np.testing.assert_allclose(prefix.distinct_knots[available - degree] + tau, executable_end,
                               atol=1e-12, rtol=0)
    x = np.linspace(0.0, executable_end - tau, 191)
    np.testing.assert_allclose(transferred(x), source(x + tau), atol=1e-12, rtol=0)
    assert np.count_nonzero(transferred.t == 0.0) == degree + 1
    # Boehm controls are convex combinations of the source controls.
    assert np.all(prefix.control_points.min(axis=0) >= controls.min(axis=0) - 1e-12)
    assert np.all(prefix.control_points.max(axis=0) <= controls.max(axis=0) + 1e-12)

    with pytest.raises(ValueError, match="future knot spans"):
        extract_left_clamped_prefix(source, tau, num_knot_spans=available + 1)


def test_prefix_of_open_left_input_is_clamped_even_at_the_domain_start():
    degree = 3
    knots = np.arange(-3.0, 13.0)
    spline = BSpline(knots, np.random.default_rng(5).normal(size=(12, 2)), degree)
    for left in (0.0, 0.5, 2.0):
        prefix = extract_left_clamped_prefix(spline, left, num_knot_spans=5)
        end = prefix.distinct_knots[prefix.num_executable_spans]
        x = np.linspace(0.0, end, 101)
        np.testing.assert_allclose(prefix.executable_spline()(x), spline(x + left),
                                   atol=1e-13, rtol=0)


def test_prefix_rejects_repeated_interior_knots_instead_of_misaligning():
    degree = 3
    knots = np.r_[[0.0] * 4, 1, 2, 2, 3, 4, 5, 6, 7, [8.0] * 4]
    spline = BSpline(knots, np.random.default_rng(6).normal(size=(len(knots) - 4, 2)), degree)
    for left in (0.0, 0.5):
        with pytest.raises(ValueError, match="repeated interior knots"):
            extract_left_clamped_prefix(spline, left, num_knot_spans=6)
    # A repeated knot exactly at the cut is clamped by subdivision and is fine.
    prefix = extract_left_clamped_prefix(spline, 2.0, num_knot_spans=5)
    x = np.linspace(0.0, prefix.distinct_knots[prefix.num_executable_spans], 61)
    np.testing.assert_allclose(prefix.executable_spline()(x), spline(x + 2.0), atol=1e-13, rtol=0)


def test_adaptive_record_transfer_separates_continuous_and_requantization_error():
    config = AdaptiveLeftBSplineConfig(
        action_dim=4, chunk_size=8, frequency_hz=10.0, degree=3,
        num_basis=7, span_length_steps=2, vocab_size=64,
    )
    encoder = create_encoder(config)
    episode = smooth_episode(48, seed=2)
    encoder.calibrate([episode])
    records = encoder.encode_episode(episode, quantize=True)
    half_bin = (encoder.quantizer.high - encoder.quantizer.low) / (2 * (config.vocab_size - 1))
    dt = config.delta_t
    for record in records[:-1:5]:
        source = record.scipy_spline()
        breaks = np.unique(source.t)
        for steps in (1, 2):  # between-knot or existing-knot cut, model dependent
            tau = steps * dt
            span = int(np.searchsorted(breaks, tau, side="right") - 1)
            available = len(breaks) - 1 - span
            if available <= config.degree:
                continue
            prefix = extract_left_clamped_prefix(source, tau, num_knot_spans=available)
            transferred = prefix.executable_spline()
            horizon = record.executable_steps * dt - tau
            x = np.linspace(0.0, horizon, 97)
            continuous_error = np.abs(transferred(x) - source(x + tau)).max()
            assert continuous_error < 1e-12

            # Requantization is a separate, bounded error with no clipping
            # because transferred controls stay inside the source hull.
            tokens = encoder.quantizer.encode(prefix.control_points)
            assert np.all(prefix.control_points >= encoder.quantizer.low - 1e-12)
            assert np.all(prefix.control_points <= encoder.quantizer.high + 1e-12)
            requantized = BSpline(transferred.t, encoder.quantizer.decode(tokens), 3, axis=0)
            requantization_error = np.abs(requantized(x) - transferred(x)).max(axis=0)
            assert np.all(requantization_error <= half_bin + 1e-12)

            # Transferring the already-dequantized source accumulates at most
            # one more half-bin after requantization.
            dequantized = BSpline(source.t, record.dequantized_control_points, 3, axis=0)
            dq_prefix = extract_left_clamped_prefix(dequantized, tau, num_knot_spans=available)
            twice = BSpline(transferred.t, encoder.quantizer.decode(
                encoder.quantizer.encode(dq_prefix.control_points)), 3, axis=0)
            assert np.all(np.abs(twice(x) - source(x + tau)).max(axis=0) <= 2 * half_bin + 1e-12)


@pytest.mark.parametrize("steps", (10, 11))
def test_uniform_left_suffix_transfer_keeps_the_original_executable_end(steps):
    encoder = create_encoder(preset("piper_chunk50", "uniform_left"))
    actions = smooth_episode(50, dim=7, seed=3)
    result = encoder.encode_chunk(actions)
    source = result.scipy_spline()
    tau = steps * result.sample_period
    breaks = np.unique(source.t)
    available = len(breaks) - 1 - int(np.searchsorted(breaks, tau, side="right") - 1)
    prefix = extract_left_clamped_prefix(source, tau, num_knot_spans=available)
    end_steps = (prefix.distinct_knots[prefix.num_executable_spans] + tau) / result.sample_period
    assert np.isclose(end_steps, result.executable_steps, rtol=0, atol=1e-9)
    x = np.linspace(0.0, (result.executable_steps - steps) * result.sample_period, 151)
    np.testing.assert_allclose(prefix.executable_spline()(x), source(x + tau), atol=1e-12, rtol=0)
