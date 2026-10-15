"""Standalone B-spline control-point fitting on a supplied knot vector."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline


FloatArray = NDArray[np.float64]


def _degree(value: Any) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or value < 1
        or value > 5
    ):
        raise ValueError("degree must be an integer between 1 and 5")
    return int(value)


def _knot_geometry(knots: Any, degree: int) -> tuple[FloatArray, int, float, float]:
    values = np.asarray(knots, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("knots must be a finite one-dimensional full knot vector")
    if np.any(np.diff(values) < 0):
        raise ValueError("knots must be nondecreasing")
    num_basis = len(values) - degree - 1
    if num_basis < degree + 1:
        raise ValueError("knot vector has too few knots for the requested degree")
    start, end = float(values[degree]), float(values[num_basis])
    if start >= end:
        raise ValueError("knot vector must define a domain with positive length")
    return values, num_basis, start, end


def _timestamps(values: Any, *, name: str = "timestamps") -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def make_clamped_knots(breakpoints: Any, *, degree: int = 3) -> FloatArray:
    """Expand distinct span boundaries into a full clamped knot vector.

    For example, cubic breakpoints ``[0, 1, 2]`` produce
    ``[0, 0, 0, 0, 1, 2, 2, 2, 2]``.
    """

    order = _degree(degree)
    breaks = _timestamps(breakpoints, name="breakpoints")
    if len(breaks) < 2 or np.any(np.diff(breaks) <= 0):
        raise ValueError("breakpoints must contain at least two increasing values")
    return np.concatenate((
        np.repeat(breaks[0], order),
        breaks,
        np.repeat(breaks[-1], order),
    ))


def make_uniform_clamped_knots(
    start: float,
    end: float,
    *,
    num_basis: int,
    degree: int = 3,
) -> FloatArray:
    """Construct a full double-clamped knot vector with uniform spans."""

    order = _degree(degree)
    if (
        not isinstance(num_basis, Integral)
        or isinstance(num_basis, (bool, np.bool_))
        or num_basis < order + 1
    ):
        raise ValueError("num_basis must be an integer of at least degree + 1")
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("start and end must be finite with start < end")
    breaks = np.linspace(float(start), float(end), int(num_basis) - order + 1)
    return make_clamped_knots(breaks, degree=order)


def make_uniform_left_clamped_knots(
    start: float,
    end: float,
    *,
    num_basis: int,
    degree: int = 3,
) -> FloatArray:
    """Construct uniform knots clamped at ``start`` and open after ``end``.

    ``[start, end]`` is the base interval.  The final ``degree`` knot spans
    extend beyond ``end`` to supply right-side basis support.
    """

    order = _degree(degree)
    if (
        not isinstance(num_basis, Integral)
        or isinstance(num_basis, (bool, np.bool_))
        or num_basis < order + 1
    ):
        raise ValueError("num_basis must be an integer of at least degree + 1")
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("start and end must be finite with start < end")
    span = (float(end) - float(start)) / (int(num_basis) - order)
    breaks = float(start) + np.arange(int(num_basis) + 1) * span
    return np.concatenate((np.repeat(float(start), order), breaks))


def bspline_design_matrix(
    timestamps: Any,
    knots: Any,
    *,
    degree: int = 3,
    extrapolate: bool = False,
) -> FloatArray:
    """Return the dense basis matrix for timestamps and a full knot vector."""

    order = _degree(degree)
    knot_values, _, start, end = _knot_geometry(knots, order)
    times = _timestamps(timestamps)
    if not extrapolate and (np.any(times < start) or np.any(times > end)):
        raise ValueError(f"timestamps must lie in the spline domain [{start}, {end}]")
    try:
        matrix = BSpline.design_matrix(
            times,
            knot_values,
            order,
            extrapolate=extrapolate,
        )
    except ValueError as exc:
        raise ValueError("cannot evaluate the supplied knot vector at the timestamps") from exc
    return np.asarray(matrix.toarray(), dtype=np.float64)


def evaluate_bspline(
    control_points: Any,
    knots: Any,
    timestamps: Any,
    *,
    degree: int = 3,
    extrapolate: bool = False,
) -> FloatArray:
    """Evaluate control points on a full knot vector at arbitrary timestamps."""

    order = _degree(degree)
    knot_values, num_basis, start, end = _knot_geometry(knots, order)
    controls = np.asarray(control_points, dtype=np.float64)
    if controls.ndim != 2 or controls.shape[0] != num_basis or not np.all(
        np.isfinite(controls)
    ):
        raise ValueError(
            f"control_points must be finite with shape ({num_basis}, action_dim)"
        )
    times = _timestamps(timestamps)
    if not extrapolate and (np.any(times < start) or np.any(times > end)):
        raise ValueError(f"timestamps must lie in the spline domain [{start}, {end}]")
    return np.asarray(
        BSpline(
            knot_values,
            controls,
            order,
            extrapolate=extrapolate,
            axis=0,
        )(times),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BSplineControlPointFit:
    """Control points and diagnostics from one multi-dimensional fit."""

    control_points: FloatArray
    knots: FloatArray
    degree: int
    timestamps: FloatArray
    actions: FloatArray
    fitted_actions: FloatArray
    weights: FloatArray | None
    regularization: float
    design_rank: int
    condition_number: float

    @property
    def num_basis(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.control_points.shape[1])

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.knots[self.degree]), float(self.knots[self.num_basis])

    @property
    def residuals(self) -> FloatArray:
        return self.fitted_actions - self.actions

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.residuals))))

    @property
    def max_absolute_error(self) -> float:
        return float(np.max(np.abs(self.residuals)))

    def scipy_spline(self, *, extrapolate: bool = False) -> BSpline:
        """Return the fitted SciPy spline."""

        return BSpline(
            self.knots,
            self.control_points,
            self.degree,
            extrapolate=extrapolate,
            axis=0,
        )

    def evaluate(self, timestamps: Any, *, extrapolate: bool = False) -> FloatArray:
        """Evaluate the fitted segment at requested timestamps."""

        return evaluate_bspline(
            self.control_points,
            self.knots,
            timestamps,
            degree=self.degree,
            extrapolate=extrapolate,
        )

    def __call__(self, timestamps: Any) -> FloatArray:
        return self.evaluate(timestamps)

    def sample(self, count: int, *, endpoint: bool = True) -> tuple[FloatArray, FloatArray]:
        """Return evenly spaced ``(timestamps, actions)`` across the full domain."""

        if (
            not isinstance(count, Integral)
            or isinstance(count, (bool, np.bool_))
            or count < 1
        ):
            raise ValueError("count must be a positive integer")
        start, end = self.domain
        times = np.linspace(start, end, int(count), endpoint=endpoint)
        return times, self.evaluate(times)


def fit_bspline(
    actions: Any,
    timestamps: Any,
    knots: Any,
    *,
    degree: int = 3,
    weights: Any | None = None,
    regularization: float = 0.0,
    require_full_rank: bool = True,
) -> BSplineControlPointFit:
    """Fit all action dimensions in parallel on a supplied full knot vector.

    The solved objective is ``||W (B C - A)||² + regularization * ||C||²``,
    where ``B`` is the B-spline design matrix, ``C`` contains the control
    points, and ``A`` contains the timestamped actions. A stable augmented
    least-squares solve is used instead of forming normal equations.
    """

    order = _degree(degree)
    times = _timestamps(timestamps)
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps used for fitting must be strictly increasing")
    values = np.asarray(actions, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[0] != len(times)
        or not values.shape[1]
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "actions must be finite with shape (len(timestamps), action_dim)"
        )
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    if not isinstance(require_full_rank, (bool, np.bool_)):
        raise TypeError("require_full_rank must be boolean")

    knot_values, num_basis, _, _ = _knot_geometry(knots, order)
    design = bspline_design_matrix(times, knot_values, degree=order)
    weight_values = None
    weighted_design = design
    weighted_actions = values
    if weights is not None:
        weight_values = np.asarray(weights, dtype=np.float64)
        if (
            weight_values.shape != (len(times),)
            or not np.all(np.isfinite(weight_values))
            or np.any(weight_values <= 0)
        ):
            raise ValueError("weights must be finite and positive with one per timestamp")
        weighted_design = design * weight_values[:, None]
        weighted_actions = values * weight_values[:, None]

    design_rank = int(np.linalg.matrix_rank(weighted_design))
    if require_full_rank and design_rank < num_basis:
        raise ValueError(
            f"B-spline design matrix is rank deficient ({design_rank} < {num_basis}); "
            "add samples/change knots or set require_full_rank=False"
        )
    solve_design = weighted_design
    solve_actions = weighted_actions
    if regularization > 0:
        solve_design = np.vstack((
            weighted_design,
            np.sqrt(regularization) * np.eye(num_basis),
        ))
        solve_actions = np.vstack((
            weighted_actions,
            np.zeros((num_basis, values.shape[1]), dtype=np.float64),
        ))
    controls = np.linalg.lstsq(solve_design, solve_actions, rcond=None)[0]
    fitted = design @ controls
    return BSplineControlPointFit(
        control_points=np.asarray(controls, dtype=np.float64).copy(),
        knots=knot_values.copy(),
        degree=order,
        timestamps=times.copy(),
        actions=values.copy(),
        fitted_actions=np.asarray(fitted, dtype=np.float64).copy(),
        weights=None if weight_values is None else weight_values.copy(),
        regularization=float(regularization),
        design_rank=design_rank,
        condition_number=float(np.linalg.cond(weighted_design)),
    )


def fit_control_points(
    actions: Any,
    timestamps: Any,
    knots: Any,
    **options: Any,
) -> FloatArray:
    """Convenience wrapper returning only the fitted control-point matrix."""

    return fit_bspline(actions, timestamps, knots, **options).control_points.copy()


fit_bspline_control_points = fit_control_points


__all__ = [
    "BSplineControlPointFit",
    "bspline_design_matrix",
    "evaluate_bspline",
    "fit_bspline",
    "fit_bspline_control_points",
    "fit_control_points",
    "make_clamped_knots",
    "make_uniform_clamped_knots",
    "make_uniform_left_clamped_knots",
]
