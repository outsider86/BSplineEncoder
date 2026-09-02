"""Reference-compatible global adaptive B-spline fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline


@dataclass(frozen=True)
class AdaptiveBSplineFit:
    spline: BSpline
    timestamps: np.ndarray
    actions: np.ndarray
    knots: np.ndarray
    control_points: np.ndarray
    degree: int
    max_abs_error: float
    weighted_squared_error: float
    converged: bool
    candidates_evaluated: int


def fit_adaptive_bspline(
    actions,
    *,
    timestamps=None,
    degree: int = 3,
    max_error: float = 0.01,
    smoothing: float = 1e-12,
    weights=None,
    max_control_points: int | None = None,
) -> AdaptiveBSplineFit:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] <= degree or values.shape[1] == 0:
        raise ValueError("actions must have shape (steps > degree, action_dim)")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must be finite")
    times = (np.arange(len(values), dtype=np.float64) if timestamps is None
             else np.asarray(timestamps, dtype=np.float64))
    if times.shape != (len(values),) or np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing with one per action")
    weight_array = None if weights is None else np.asarray(weights, dtype=np.float64)
    if weight_array is not None and (
        weight_array.shape != (len(values),) or np.any(weight_array <= 0)
    ):
        raise ValueError("weights must be positive with one value per action")
    if degree < 1 or degree > 5 or max_error <= 0 or smoothing < 0:
        raise ValueError("invalid FITPACK configuration")
    if max_control_points is not None and max_control_points < degree + 1:
        raise ValueError("max_control_points must be at least degree + 1")

    candidates: Iterable[np.ndarray] = generate_knots(
        times,
        values,
        w=weight_array,
        k=degree,
        s=smoothing,
        nest=None if max_control_points is None else max_control_points + degree + 1,
    )
    selected = selected_knots = None
    selected_max = selected_weighted = float("inf")
    converged = False
    evaluated = 0
    for candidate in candidates:
        knots = np.asarray(candidate, dtype=np.float64)
        spline = make_lsq_spline(times, values, knots, k=degree, w=weight_array)
        residual = np.asarray(spline(times) - values, dtype=np.float64)
        candidate_max = float(np.max(np.abs(residual)))
        candidate_weighted = float(np.sum(
            residual**2 if weight_array is None
            else (weight_array[:, None] * residual) ** 2
        ))
        selected, selected_knots = spline, knots
        selected_max, selected_weighted = candidate_max, candidate_weighted
        evaluated += 1
        if candidate_max < max_error:
            converged = True
            break
    if selected is None:
        raise RuntimeError("FITPACK generated no candidate knot vector")
    return AdaptiveBSplineFit(
        spline=selected,
        timestamps=times.copy(),
        actions=values.copy(),
        knots=selected_knots.copy(),
        control_points=np.asarray(selected.c, dtype=np.float64).copy(),
        degree=degree,
        max_abs_error=selected_max,
        weighted_squared_error=selected_weighted,
        converged=converged,
        candidates_evaluated=evaluated,
    )


def encode_knot_durations(
    distinct_knots,
    *,
    delta_t: float,
    tolerance: float = 1e-8,
    max_duration: int | None = None,
) -> np.ndarray:
    knots = np.asarray(distinct_knots, dtype=np.float64)
    if knots.ndim != 1 or len(knots) < 2 or delta_t <= 0:
        raise ValueError("invalid knots or delta_t")
    if not np.isclose(knots[0], 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("first local knot must be zero")
    gaps = np.diff(knots) / delta_t
    rounded = np.rint(gaps)
    if not np.allclose(gaps, rounded, rtol=0.0, atol=tolerance):
        raise ValueError("knot gaps are not aligned to integer timesteps")
    durations = rounded.astype(np.int64)
    if np.any(durations < 1):
        raise ValueError("knot durations must be positive")
    if max_duration is not None and np.any(durations > max_duration):
        raise ValueError("knot duration exceeds configured maximum")
    return durations


__all__ = ["AdaptiveBSplineFit", "encode_knot_durations", "fit_adaptive_bspline"]
