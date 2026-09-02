"""Exact B-spline knot insertion, subdivision, and interval extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline

from .result import SplineParameters


@dataclass(frozen=True)
class KnotInsertion:
    spline: BSpline
    matrix: np.ndarray


@dataclass(frozen=True)
class LeftClampedPrefix:
    distinct_knots: np.ndarray
    control_points: np.ndarray
    degree: int

    @property
    def num_knot_spans(self) -> int:
        return len(self.distinct_knots) - 1

    @property
    def num_executable_spans(self) -> int:
        return self.num_knot_spans - self.degree

    def executable_spline(self) -> BSpline:
        knots = np.concatenate((
            np.repeat(self.distinct_knots[0], self.degree + 1),
            self.distinct_knots[1:],
        ))
        return BSpline(knots, self.control_points, self.degree,
                       extrapolate=False, axis=0)


def knot_multiplicity(knots: np.ndarray, value: float) -> int:
    return int(np.count_nonzero(np.asarray(knots) == value))


def insert_knot_once(spline: BSpline, value: float) -> KnotInsertion:
    knots = np.asarray(spline.t, dtype=np.float64)
    controls = np.asarray(spline.c, dtype=np.float64)
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if not start < value < end:
        raise ValueError(f"knot insertion value must lie inside ({start}, {end})")
    multiplicity = knot_multiplicity(knots, value)
    if multiplicity >= degree:
        raise ValueError("interior knot multiplicity must be below the degree")

    count = controls.shape[0]
    span = int(np.searchsorted(knots, value, side="right") - 1)
    matrix = np.zeros((count + 1, count), dtype=np.float64)
    first_blended = span - degree + 1
    last_blended = span - multiplicity
    for row in range(0, span - degree + 1):
        matrix[row, row] = 1.0
    for row in range(first_blended, last_blended + 1):
        denominator = knots[row + degree] - knots[row]
        alpha = (value - knots[row]) / denominator
        matrix[row, row - 1] = 1.0 - alpha
        matrix[row, row] = alpha
    for row in range(last_blended + 1, count + 1):
        matrix[row, row - 1] = 1.0

    refined = BSpline(
        np.insert(knots, span + 1, value),
        matrix @ controls,
        degree,
        extrapolate=spline.extrapolate,
        axis=0,
    )
    return KnotInsertion(refined, matrix)


def insert_knot(spline: BSpline, value: float, count: int = 1) -> KnotInsertion:
    if count < 0:
        raise ValueError("count must be non-negative")
    composite = np.eye(np.asarray(spline.c).shape[0], dtype=np.float64)
    refined = spline
    for _ in range(count):
        insertion = insert_knot_once(refined, value)
        composite = insertion.matrix @ composite
        refined = insertion.spline
    return KnotInsertion(refined, composite)


def subdivide(spline: BSpline, value: float) -> tuple[BSpline, BSpline]:
    """Split at an interior parameter without changing either curve."""

    knots = np.asarray(spline.t, dtype=np.float64)
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if not start < value < end:
        raise ValueError(f"subdivision value must lie inside ({start}, {end})")
    insertion_count = degree - knot_multiplicity(knots, value)
    refined = insert_knot(spline, value, insertion_count).spline
    refined_knots = np.asarray(refined.t, dtype=np.float64)
    controls = np.asarray(refined.c, dtype=np.float64)
    last_value = int(np.flatnonzero(refined_knots == value)[-1])
    shared_control = last_value - degree
    left = BSpline(
        np.concatenate((refined_knots[: last_value + 1], [value])),
        controls[: shared_control + 1],
        degree,
        extrapolate=False,
        axis=0,
    )
    right = BSpline(
        np.concatenate(([value], refined_knots[last_value - degree + 1 :])),
        controls[shared_control:],
        degree,
        extrapolate=False,
        axis=0,
    )
    return left, right


def _local_support_spline(spline: BSpline, left: float, right: float) -> BSpline:
    knots = np.asarray(spline.t, dtype=np.float64)
    controls = np.asarray(spline.c, dtype=np.float64)
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    left_span = degree if left == start else int(
        np.searchsorted(knots, left, side="left") - 1
    )
    right_span = len(controls) - 1 if right == end else int(
        np.searchsorted(knots, right, side="right") - 1
    )
    first_control = left_span - degree
    last_control = right_span
    return BSpline(
        knots[first_control : last_control + degree + 2],
        controls[first_control : last_control + 1],
        degree,
        extrapolate=False,
        axis=0,
    )


def extract_interval(spline: BSpline, left: float, right: float) -> BSpline:
    """Extract and zero-origin an exact double-clamped interval."""

    knots = np.asarray(spline.t, dtype=np.float64)
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if left < start or right > end or left >= right:
        raise ValueError(f"interval [{left}, {right}] must lie in [{start}, {end}]")
    retained = _local_support_spline(spline, left, right)
    if left > start:
        _, retained = subdivide(retained, left)
    retained_end = float(retained.t[-degree - 1])
    if right < retained_end:
        retained, _ = subdivide(retained, right)
    return BSpline(
        np.asarray(retained.t, dtype=np.float64) - left,
        np.asarray(retained.c, dtype=np.float64).copy(),
        degree,
        extrapolate=False,
        axis=0,
    )


def extract_left_clamped_prefix(
    spline: BSpline,
    left: float,
    *,
    num_knot_spans: int,
) -> LeftClampedPrefix:
    """Subdivide at ``left`` and retain fixed-size causal right context."""

    degree = int(spline.k)
    if num_knot_spans <= degree:
        raise ValueError("num_knot_spans must exceed degree")
    knots = np.asarray(spline.t, dtype=np.float64)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if left < start or left >= end:
        raise ValueError(f"left boundary must lie in [{start}, {end})")
    breaks = np.unique(knots)
    span = int(np.searchsorted(breaks, left, side="right") - 1)
    final_break = span + num_knot_spans
    if final_break >= len(breaks):
        raise ValueError("not enough future knot spans")
    support_right = float(breaks[final_break])
    retained = _local_support_spline(spline, left, support_right)
    if left > float(retained.t[degree]):
        _, retained = subdivide(retained, left)
    controls = np.asarray(retained.c, dtype=np.float64)
    if len(controls) < num_knot_spans:
        raise RuntimeError("left subdivision returned too few controls")
    distinct = np.concatenate(([left], breaks[span + 1 : final_break + 1]))
    if len(distinct) != num_knot_spans + 1:
        raise RuntimeError("left prefix has an unexpected knot-span count")
    return LeftClampedPrefix(
        distinct_knots=distinct - left,
        control_points=controls[:num_knot_spans].copy(),
        degree=degree,
    )


def subdivide_parameters(
    parameters: SplineParameters,
    step: int,
) -> tuple[SplineParameters, SplineParameters]:
    """High-level exact subdivision of a double-clamped result.

    The split is specified in native action steps.  Quantized tokens are not
    propagated because knot insertion changes the control-point calibration
    geometry; both returned objects remain lossless continuous splines.
    """

    if step <= 0 or step >= parameters.executable_steps:
        raise ValueError("step must be strictly inside the executable horizon")
    split_time = step * parameters.sample_period
    left, right = subdivide(parameters.scipy_spline(), split_time)
    left_result = SplineParameters(
        control_points=np.asarray(left.c),
        knots=np.asarray(left.t),
        degree=left.k,
        sample_period=parameters.sample_period,
        executable_steps=step,
        tokenizer_id=parameters.tokenizer_id,
        observation_index=parameters.observation_index,
        metadata={**parameters.metadata, "subdivision": "left", "split_step": step},
    )
    right_result = SplineParameters(
        control_points=np.asarray(right.c),
        knots=np.asarray(right.t) - split_time,
        degree=right.k,
        sample_period=parameters.sample_period,
        executable_steps=parameters.executable_steps - step,
        tokenizer_id=parameters.tokenizer_id,
        observation_index=(None if parameters.observation_index is None
                           else parameters.observation_index + step),
        metadata={**parameters.metadata, "subdivision": "right", "split_step": step},
    )
    return left_result, right_result


__all__ = [
    "KnotInsertion", "LeftClampedPrefix", "extract_interval",
    "extract_left_clamped_prefix", "insert_knot", "insert_knot_once",
    "knot_multiplicity", "subdivide", "subdivide_parameters",
]
