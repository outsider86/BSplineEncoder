import numpy as np
import pytest
from scipy.interpolate import BSpline

from spline_encoder import (
    bspline_design_matrix,
    evaluate_bspline,
    fit_bspline,
    fit_control_points,
    make_clamped_knots,
    make_uniform_clamped_knots,
    make_uniform_left_clamped_knots,
)


def test_knot_builders_produce_expected_full_vectors():
    actual = make_clamped_knots([0.0, 1.0, 2.5], degree=3)
    np.testing.assert_array_equal(actual, [0, 0, 0, 0, 1, 2.5, 2.5, 2.5, 2.5])
    uniform = make_uniform_clamped_knots(0.0, 4.0, num_basis=7, degree=3)
    assert len(uniform) == 11
    np.testing.assert_array_equal(uniform[:4], 0.0)
    np.testing.assert_array_equal(uniform[-4:], 4.0)
    np.testing.assert_allclose(np.unique(uniform), [0, 1, 2, 3, 4])
    left = make_uniform_left_clamped_knots(
        0.0, 8.0, num_basis=7, degree=3
    )
    np.testing.assert_array_equal(left, [0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14])
    assert (left[3], left[7]) == (0.0, 8.0)


@pytest.mark.parametrize("degree", range(1, 6))
def test_fit_control_points_recovers_known_multidimensional_spline(degree):
    rng = np.random.default_rng(4000 + degree)
    num_basis = degree + 5
    knots = make_uniform_clamped_knots(
        -0.25, 2.75, num_basis=num_basis, degree=degree
    )
    expected = rng.normal(size=(num_basis, 4))
    timestamps = np.sort(rng.uniform(-0.24, 2.74, 40))
    timestamps = np.r_[knots[degree], timestamps, knots[num_basis]]
    actions = BSpline(knots, expected, degree, axis=0)(timestamps)
    actual = fit_control_points(actions, timestamps, knots, degree=degree)
    np.testing.assert_allclose(actual, expected, atol=5e-13, rtol=0)


def test_fit_object_evaluates_samples_and_reports_diagnostics():
    knots = make_clamped_knots([0.0, 0.5, 1.5, 2.0], degree=2)
    timestamps = np.linspace(0.0, 2.0, 31)
    actions = np.stack((
        np.sin(timestamps),
        timestamps**2,
        0.5 * timestamps - 1.0,
    ), axis=1)
    fit = fit_bspline(actions, timestamps, knots, degree=2)
    assert fit.control_points.shape == (5, 3)
    assert fit.design_rank == fit.num_basis
    assert fit.domain == (0.0, 2.0)
    np.testing.assert_allclose(fit(timestamps), fit.fitted_actions, atol=1e-14, rtol=0)
    assert fit.rmse >= 0
    assert fit.max_absolute_error >= 0
    sample_times, samples = fit.sample(51)
    assert sample_times.shape == (51,)
    assert samples.shape == (51, 3)
    np.testing.assert_allclose(samples, fit.scipy_spline()(sample_times))


def test_vectorized_fit_matches_one_independent_solve_per_action_dimension():
    rng = np.random.default_rng(12)
    knots = make_uniform_clamped_knots(0.0, 3.0, num_basis=8, degree=3)
    timestamps = np.linspace(0.0, 3.0, 37)
    actions = rng.normal(size=(len(timestamps), 6))
    vectorized = fit_control_points(actions, timestamps, knots)
    independent = np.column_stack([
        fit_control_points(actions[:, index:index + 1], timestamps, knots)[:, 0]
        for index in range(actions.shape[1])
    ])
    np.testing.assert_allclose(vectorized, independent, atol=2e-15, rtol=0)


def test_weighted_regularized_fit_matches_augmented_least_squares():
    rng = np.random.default_rng(23)
    knots = make_uniform_clamped_knots(0.0, 2.0, num_basis=6, degree=3)
    timestamps = np.linspace(0.0, 2.0, 25)
    actions = rng.normal(size=(25, 2))
    weights = np.linspace(0.5, 2.0, len(timestamps))
    regularization = 0.125
    actual = fit_control_points(
        actions,
        timestamps,
        knots,
        weights=weights,
        regularization=regularization,
    )
    design = BSpline.design_matrix(timestamps, knots, 3).toarray()
    augmented_design = np.vstack((
        design * weights[:, None],
        np.sqrt(regularization) * np.eye(6),
    ))
    augmented_actions = np.vstack((
        actions * weights[:, None],
        np.zeros((6, 2)),
    ))
    expected = np.linalg.lstsq(augmented_design, augmented_actions, rcond=None)[0]
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)


def test_design_and_evaluation_helpers_match_scipy():
    knots = make_uniform_clamped_knots(1.0, 5.0, num_basis=7, degree=3)
    timestamps = np.linspace(1.0, 5.0, 17)
    controls = np.arange(21, dtype=np.float64).reshape(7, 3)
    design = bspline_design_matrix(timestamps, knots)
    np.testing.assert_allclose(
        design, BSpline.design_matrix(timestamps, knots, 3).toarray()
    )
    np.testing.assert_allclose(
        evaluate_bspline(controls, knots, timestamps),
        BSpline(knots, controls, 3, axis=0)(timestamps),
    )


def test_rank_deficiency_is_explicit_but_can_be_opted_into():
    knots = make_uniform_clamped_knots(0.0, 1.0, num_basis=8, degree=3)
    timestamps = np.linspace(0.0, 1.0, 5)
    actions = np.stack((timestamps, timestamps**2), axis=1)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_control_points(actions, timestamps, knots)
    fit = fit_bspline(
        actions,
        timestamps,
        knots,
        require_full_rank=False,
        regularization=1e-4,
    )
    assert fit.design_rank < fit.num_basis
    assert np.all(np.isfinite(fit.control_points))


@pytest.mark.parametrize(
    "timestamps,actions,message",
    (
        ([0.0, 0.5, 0.5, 1.0], np.zeros((4, 1)), "strictly increasing"),
        ([0.0, 0.5, 1.0], np.zeros((2, 1)), "shape"),
        ([-0.1, 0.5, 1.0], np.zeros((3, 1)), "domain"),
    ),
)
def test_fit_rejects_invalid_sample_geometry(timestamps, actions, message):
    knots = make_uniform_clamped_knots(0.0, 1.0, num_basis=4, degree=3)
    with pytest.raises(ValueError, match=message):
        fit_control_points(actions, timestamps, knots)
