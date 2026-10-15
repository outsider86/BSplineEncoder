"""Independent-reference checks for basis evaluation, insertion, and cutting.

References used here do not call package code: SciPy's ``BSpline`` and
``design_matrix``, FITPACK knot insertion (``scipy.interpolate.insert``),
``PPoly.from_spline`` piecewise polynomials for one-sided derivatives, and
LAPACK least squares (``scipy.linalg.lstsq``).
"""

import numpy as np
import pytest
import scipy.linalg
from scipy.interpolate import BSpline, PPoly, insert as fitpack_insert

from spline_encoder import (
    bspline_design_matrix,
    extract_interval,
    fit_bspline,
    insert_knot,
    make_uniform_left_clamped_knots,
    subdivide,
)
from spline_encoder.encoder import _design

DEGREES = range(1, 6)


def nonuniform_clamped(degree, spans=8, seed=0, start=0.0):
    rng = np.random.default_rng(seed)
    breaks = start + np.r_[0.0, np.cumsum(rng.uniform(0.25, 1.75, spans))]
    knots = np.r_[np.repeat(breaks[0], degree), breaks, np.repeat(breaks[-1], degree)]
    return knots, breaks


def random_spline(knots, degree, dim=3, seed=1):
    controls = np.random.default_rng(seed).normal(size=(len(knots) - degree - 1, dim))
    return BSpline(knots, controls, degree, extrapolate=False, axis=0)


def one_sided(spline, x, order, side):
    """Piecewise-polynomial derivative limit, independent of package code."""

    t, c, k = spline.t, np.asarray(spline.c), int(spline.k)
    values = []
    for column in range(c.shape[1]):
        pp = PPoly.from_spline((t, c[:, column], k))
        index = int(np.searchsorted(pp.x, x, side="left" if side == "left" else "right") - 1)
        poly = np.polyder(pp.c[:, index], order) if order else pp.c[:, index]
        values.append(np.polyval(poly, x - pp.x[index]))
    return np.asarray(values)


@pytest.mark.parametrize("degree", DEGREES)
def test_basis_matches_scipy_on_nonuniform_and_right_open_knots(degree):
    knots, breaks = nonuniform_clamped(degree, seed=degree)
    samples = np.r_[np.linspace(breaks[0], breaks[-1], 97), breaks]
    expected = BSpline.design_matrix(samples, knots, degree).toarray()
    np.testing.assert_allclose(_design(samples, knots, degree), expected, atol=1e-15, rtol=0)
    np.testing.assert_allclose(
        bspline_design_matrix(samples, knots, degree=degree), expected, atol=0, rtol=0
    )

    open_knots = make_uniform_left_clamped_knots(0.0, 10.0, num_basis=degree + 6, degree=degree)
    samples = np.r_[np.linspace(0.0, 10.0, 81), open_knots[(open_knots >= 0) & (open_knots <= 10)]]
    expected = BSpline.design_matrix(samples, open_knots, degree).toarray()
    np.testing.assert_allclose(_design(samples, open_knots, degree), expected, atol=1e-15, rtol=0)
    # Partition of unity holds on the whole base interval of a right-open vector.
    np.testing.assert_allclose(expected.sum(axis=1), 1.0, atol=1e-14, rtol=0)


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("location", ("between", "existing"))
def test_knot_insertion_matches_fitpack(degree, location):
    if degree == 1 and location == "existing":
        pytest.skip("degree-1 insertion at a simple knot would exceed multiplicity < degree")
    knots, breaks = nonuniform_clamped(degree, seed=10 + degree)
    spline = random_spline(knots, degree, seed=20 + degree)
    value = breaks[4] if location == "existing" else 0.37 * breaks[3] + 0.63 * breaks[4]
    refined = insert_knot(spline, value).spline
    for column in range(spline.c.shape[1]):
        t_ref, c_ref, _ = fitpack_insert(value, (knots, spline.c[:, column], degree))
        np.testing.assert_array_equal(refined.t, t_ref)
        np.testing.assert_allclose(
            refined.c[:, column], np.asarray(c_ref)[: len(refined.c)], atol=1e-14, rtol=0
        )


@pytest.mark.parametrize("degree", DEGREES)
def test_known_nonuniform_and_right_open_splines_are_recovered_by_fitting(degree):
    rng = np.random.default_rng(300 + degree)
    for knots in (
        nonuniform_clamped(degree, seed=degree)[0],
        make_uniform_left_clamped_knots(0.0, 3.0, num_basis=degree + 7, degree=degree),
    ):
        num_basis = len(knots) - degree - 1
        start, end = knots[degree], knots[num_basis]
        expected = rng.normal(size=(num_basis, 2))
        times = np.r_[start, np.sort(rng.uniform(start, end, 12 * num_basis)), end]
        actions = BSpline(knots, expected, degree, axis=0)(times)
        weights = rng.uniform(0.5, 2.0, len(times))
        fit = fit_bspline(actions, times, knots, degree=degree, weights=weights)
        design = BSpline.design_matrix(times, knots, degree).toarray()
        reference = scipy.linalg.lstsq(
            design * weights[:, None], actions * weights[:, None], lapack_driver="gelsd"
        )[0]
        tolerance = 1e-13 * max(1.0, fit.condition_number)
        np.testing.assert_allclose(fit.control_points, reference, atol=tolerance, rtol=0)
        np.testing.assert_allclose(fit.control_points, expected, atol=tolerance, rtol=0)


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("location", ("between", "existing", "double"))
def test_subdivision_preserves_curve_and_one_sided_derivatives(degree, location):
    knots, breaks = nonuniform_clamped(degree, seed=40 + degree)
    if location == "double":
        if degree < 2:
            pytest.skip("a double interior knot needs degree >= 2")
        knots = np.sort(np.r_[knots, breaks[4]])
    spline = random_spline(knots, degree, seed=50 + degree)
    cut = 0.41 * breaks[3] + 0.59 * breaks[4] if location == "between" else breaks[4]
    left, right = subdivide(spline, cut)

    assert np.all(left.t[-degree - 1:] == cut) and np.all(right.t[: degree + 1] == cut)
    x_left = np.linspace(breaks[0], cut, 211)
    x_right = np.linspace(cut, breaks[-1], 211)
    np.testing.assert_allclose(left(x_left), spline(x_left), atol=2e-13, rtol=0)
    np.testing.assert_allclose(right(x_right), spline(x_right), atol=2e-13, rtol=0)
    for order in range(degree + 1):
        scale = 1e-11 * 10 ** order
        np.testing.assert_allclose(
            one_sided(left, cut, order, "left"), one_sided(spline, cut, order, "left"),
            atol=scale, rtol=1e-11,
        )
        np.testing.assert_allclose(
            one_sided(right, cut, order, "right"), one_sided(spline, cut, order, "right"),
            atol=scale, rtol=1e-11,
        )
    # The pieces are as continuous as the source at the cut: C^(degree-m).
    multiplicity = {"between": 0, "existing": 1, "double": 2}[location]
    for order in range(degree - multiplicity + 1):
        np.testing.assert_allclose(
            one_sided(left, cut, order, "left"), one_sided(right, cut, order, "right"),
            atol=1e-10 * 10 ** order, rtol=1e-10,
        )


@pytest.mark.parametrize("degree", DEGREES)
@pytest.mark.parametrize("layout", ("clamped", "right_open", "open_uniform"))
def test_extract_interval_is_double_clamped_at_every_boundary(degree, layout):
    if layout == "clamped":
        knots, breaks = nonuniform_clamped(degree, seed=60 + degree)
    elif layout == "right_open":
        knots = make_uniform_left_clamped_knots(0.0, 6.0, num_basis=degree + 6, degree=degree)
    else:
        knots = np.arange(-degree, degree + 12, dtype=np.float64)
    spline = random_spline(knots, degree, seed=70 + degree)
    start, end = float(knots[degree]), float(knots[len(spline.c)])
    inner = (0.7 * start + 0.3 * end, 0.35 * start + 0.65 * end)
    for left, right in ((start, end), (start, inner[1]), (inner[0], end), inner):
        interval = extract_interval(spline, left, right)
        width = right - left
        assert np.count_nonzero(interval.t == 0.0) == degree + 1
        assert np.count_nonzero(np.isclose(interval.t, width, rtol=0, atol=1e-12)) == degree + 1
        assert len(interval.t) == len(interval.c) + degree + 1
        x = np.linspace(0.0, width, 157)
        np.testing.assert_allclose(interval(x), spline(x + left), atol=5e-13, rtol=0)
        for order in range(degree + 1):
            np.testing.assert_allclose(
                one_sided(interval, 0.0, order, "right"),
                one_sided(spline, left, order, "right"),
                atol=1e-10 * 10 ** order, rtol=1e-10,
            )
