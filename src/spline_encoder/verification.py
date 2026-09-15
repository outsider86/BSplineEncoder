"""Small, JSON-friendly checks for encoded spline results."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline

from .result import SplineParameters


@dataclass(frozen=True)
class ReconstructionMetrics:
    """Error statistics between two ``(steps, action_dim)`` arrays."""

    sample_count: int
    action_dim: int
    mae: float
    rmse: float
    max_absolute_error: float
    mae_per_action_dimension: tuple[float, ...]
    rmse_per_action_dimension: tuple[float, ...]
    max_absolute_error_per_action_dimension: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a representation that can be written directly as JSON."""

        payload = asdict(self)
        for key in (
            "mae_per_action_dimension",
            "rmse_per_action_dimension",
            "max_absolute_error_per_action_dimension",
        ):
            payload[key] = list(payload[key])
        return payload


def _action_matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must have shape (steps, action_dim)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def reconstruction_metrics(reference: Any, reconstructed: Any) -> ReconstructionMetrics:
    """Compute deterministic aggregate and per-dimension reconstruction errors."""

    target = _action_matrix(reference, name="reference")
    prediction = _action_matrix(reconstructed, name="reconstructed")
    if prediction.shape != target.shape:
        raise ValueError(
            "reference and reconstructed actions must have the same shape, "
            f"got {target.shape} and {prediction.shape}"
        )
    absolute = np.abs(prediction - target)
    squared = np.square(prediction - target)
    return ReconstructionMetrics(
        sample_count=int(target.shape[0]),
        action_dim=int(target.shape[1]),
        mae=float(absolute.mean()),
        rmse=float(np.sqrt(squared.mean())),
        max_absolute_error=float(absolute.max()),
        mae_per_action_dimension=tuple(absolute.mean(axis=0).tolist()),
        rmse_per_action_dimension=tuple(np.sqrt(squared.mean(axis=0)).tolist()),
        max_absolute_error_per_action_dimension=tuple(absolute.max(axis=0).tolist()),
    )


def verify_spline_parameters(
    parameters: SplineParameters,
    *,
    absolute_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Check geometry, duration round-trip, partition of unity, and finite decode.

    This intentionally evaluates invariants that do not depend on how the
    encoder fitted its controls, so it can catch malformed stored or
    reconstructed results without blessing the implementation under test.
    """

    if not np.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be finite and non-negative")
    knots = np.asarray(parameters.knots, dtype=np.float64)
    controls = np.asarray(parameters.control_points, dtype=np.float64)
    degree = int(parameters.degree)
    native_times = np.arange(parameters.executable_steps) * parameters.sample_period
    domain_start = float(knots[degree])
    domain_end = float(knots[len(controls)])
    decoded = parameters.decode()
    unity = BSpline(
        knots,
        np.ones(len(controls), dtype=np.float64),
        degree,
        extrapolate=False,
    )(native_times)

    checks = {
        "knot_control_cardinality": len(knots) == len(controls) + degree + 1,
        "nondecreasing_knots": bool(np.all(np.diff(knots) >= 0)),
        "zero_origin": bool(np.isclose(domain_start, 0.0, rtol=0.0, atol=absolute_tolerance)),
        "executable_samples_inside_domain": bool(
            native_times[0] >= domain_start - absolute_tolerance
            and native_times[-1] <= domain_end + absolute_tolerance
        ),
        "left_clamped": int(np.count_nonzero(np.isclose(
            knots, domain_start, rtol=0.0, atol=absolute_tolerance
        ))) >= degree + 1,
        "partition_of_unity": bool(np.allclose(
            unity, 1.0, rtol=0.0, atol=absolute_tolerance
        )),
        "finite_native_decode": bool(
            decoded.shape == (parameters.executable_steps, parameters.action_dim)
            and np.all(np.isfinite(decoded))
        ),
    }
    duration_round_trip: bool | None = None
    if parameters.duration_steps is not None:
        distinct_steps = np.r_[0, np.cumsum(parameters.duration_steps)]
        duration_knots = np.r_[np.zeros(degree + 1), distinct_steps[1:]]
        duration_round_trip = bool(np.allclose(
            duration_knots,
            parameters.knot_steps,
            rtol=0.0,
            atol=absolute_tolerance,
        ))
        checks["duration_knot_round_trip"] = duration_round_trip

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "degree": degree,
        "num_basis": parameters.num_basis,
        "action_dim": parameters.action_dim,
        "executable_steps": parameters.executable_steps,
        "domain_seconds": [domain_start, domain_end],
        "duration_knot_round_trip": duration_round_trip,
    }


def verify_reconstruction(
    parameters: SplineParameters,
    reference: Any,
    *,
    max_absolute_error: float | None = None,
    dequantized: bool = False,
) -> dict[str, Any]:
    """Verify one result against its endpoint-exclusive native action target.

    ``reference`` must contain exactly ``parameters.executable_steps`` rows.
    When ``max_absolute_error`` is omitted, ``passed`` reflects spline
    invariants only and the error statistics remain descriptive.
    """

    target = _action_matrix(reference, name="reference")
    expected = (parameters.executable_steps, parameters.action_dim)
    if target.shape != expected:
        raise ValueError(f"reference must have shape {expected}, got {target.shape}")
    if max_absolute_error is not None and (
        not np.isfinite(max_absolute_error) or max_absolute_error < 0
    ):
        raise ValueError("max_absolute_error must be finite and non-negative")
    decoded = parameters.decode(dequantized=dequantized)
    metrics = reconstruction_metrics(target, decoded)
    invariants = verify_spline_parameters(parameters)
    error_within_tolerance = (
        None
        if max_absolute_error is None
        else metrics.max_absolute_error <= max_absolute_error
    )
    return {
        "passed": bool(
            invariants["passed"]
            and (error_within_tolerance is None or error_within_tolerance)
        ),
        "representation": "dequantized" if dequantized else "continuous",
        "max_absolute_error_tolerance": max_absolute_error,
        "error_within_tolerance": error_within_tolerance,
        "metrics": metrics.to_dict(),
        "invariants": invariants,
    }


__all__ = [
    "ReconstructionMetrics",
    "reconstruction_metrics",
    "verify_reconstruction",
    "verify_spline_parameters",
]
