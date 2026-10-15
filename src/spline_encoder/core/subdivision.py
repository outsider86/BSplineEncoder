"""Exact B-spline knot insertion, subdivision, and interval extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import BSpline

if TYPE_CHECKING:  # result objects live in encoding; core never imports it eagerly
    from ..encoding.result import SplineParameters


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
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if not start < value < end:
        raise ValueError(f"knot insertion value must lie inside ({start}, {end})")
    if knot_multiplicity(knots, value) >= degree:
        raise ValueError("interior knot multiplicity must be below the degree")
    return _boehm_insert(spline, value)


def _boehm_insert(spline: BSpline, value: float) -> KnotInsertion:
    """Insert one knot anywhere in the closed base interval ``[t[k], t[n]]``.

    For interior values this is arithmetically identical to the historical
    :func:`insert_knot_once`.  At the right domain end of a right-open knot
    vector the active span is the last base span, and only copies of
    ``value`` inside that span's knot prefix count toward the multiplicity.
    """

    knots = np.asarray(spline.t, dtype=np.float64)
    controls = np.asarray(spline.c, dtype=np.float64)
    degree = int(spline.k)
    count = controls.shape[0]
    span = min(int(np.searchsorted(knots, value, side="right") - 1), count - 1)
    multiplicity = knot_multiplicity(knots[: span + 1], value)
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


def _clamped_piece(spline: BSpline, value: float, side: str) -> BSpline:
    """Return the exact piece on one ``side`` of ``value``, clamped at ``value``.

    Unlike :func:`subdivide`, ``value`` may equal a base-domain endpoint.  This
    clamps open (non-repeated) boundary knots, for example the right end of a
    left-clamped/right-open action spline.  An already clamped boundary is
    returned with identical knots and controls.
    """

    degree = int(spline.k)
    refined = spline
    while knot_multiplicity(refined.t, value) < degree:
        refined = _boehm_insert(refined, value).spline
    knots = np.asarray(refined.t, dtype=np.float64)
    controls = np.asarray(refined.c, dtype=np.float64)
    hits = np.flatnonzero(knots == value)
    if side == "right":
        last = int(hits[-1])
        piece_knots = np.concatenate((np.repeat(value, degree + 1), knots[last + 1 :]))
        piece_controls = controls[last - degree :]
    else:
        first = int(hits[0])
        piece_knots = np.concatenate((knots[:first], np.repeat(value, degree + 1)))
        piece_controls = controls[:first]
    return BSpline(piece_knots, piece_controls, degree, extrapolate=False, axis=0)


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
    """Extract and zero-origin an exact double-clamped interval.

    Both returned boundaries have multiplicity ``degree + 1`` even when
    ``left``/``right`` coincide with an open (unclamped) end of the input
    knot vector.
    """

    knots = np.asarray(spline.t, dtype=np.float64)
    degree = int(spline.k)
    start, end = float(knots[degree]), float(knots[-degree - 1])
    if left < start or right > end or left >= right:
        raise ValueError(f"interval [{left}, {right}] must lie in [{start}, {end}]")
    retained = _local_support_spline(spline, left, right)
    if left > start:
        _, retained = subdivide(retained, left)
    else:
        retained = _clamped_piece(retained, left, "right")
    retained_end = float(retained.t[-degree - 1])
    if right < retained_end:
        retained, _ = subdivide(retained, right)
    else:
        retained = _clamped_piece(retained, right, "left")
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
    """Subdivide at ``left`` and retain fixed-size causal right context.

    The input may be clamped or open at its left domain end.  Knots strictly
    between ``left`` and the last retained break must be simple: one
    control-duration pair per span cannot represent a repeated interior knot.
    """

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
    interior = knots[(knots > left) & (knots < support_right)]
    if len(np.unique(interior)) != len(interior):
        raise ValueError(
            "left-clamped prefixes require simple knots between left and the "
            "retained support; repeated interior knots are unsupported"
        )
    retained = _local_support_spline(spline, left, support_right)
    if left > float(retained.t[degree]):
        _, retained = subdivide(retained, left)
    else:
        retained = _clamped_piece(retained, left, "right")
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

    from ..encoding.result import SplineParameters

    spline = parameters.scipy_spline()
    split_time = step * parameters.sample_period
    start, end = float(spline.t[spline.k]), float(spline.t[parameters.num_basis])
    # With inclusive_domain the last native sample sits on the domain end, so
    # the final step is not an interior split point.
    if (
        step <= 0
        or step >= parameters.executable_steps
        or not start < split_time < end
    ):
        raise ValueError("step must be strictly inside the executable horizon")
    left, right = subdivide(spline, split_time)
    left_metadata = {**parameters.metadata, "subdivision": "left", "split_step": step}
    if left_metadata.get("boundary_condition") == "left_clamped_right_open":
        # Subdivision clamps the cut, so the left piece is double-clamped.
        left_metadata["boundary_condition"] = "double_clamped"
    left_result = SplineParameters(
        control_points=np.asarray(left.c),
        knots=np.asarray(left.t),
        degree=left.k,
        sample_period=parameters.sample_period,
        executable_steps=step,
        tokenizer_id=parameters.tokenizer_id,
        observation_index=parameters.observation_index,
        metadata=left_metadata,
    )
    right_knots = np.asarray(right.t, dtype=np.float64) - split_time
    if parameters.metadata.get("inclusive_domain", False):
        # Inclusive layouts sample the domain end itself.  ``end - split_time``
        # can round one ulp below ``(steps - 1) * sample_period``, which made
        # the right child's final native sample decode to NaN.
        horizon = (parameters.executable_steps - step - 1) * parameters.sample_period
        at_end = np.asarray(right.t) == end
        if np.isclose(horizon, end - split_time, rtol=0.0,
                      atol=1e-9 * parameters.sample_period):
            right_knots[at_end] = horizon
    right_result = SplineParameters(
        control_points=np.asarray(right.c),
        knots=right_knots,
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
