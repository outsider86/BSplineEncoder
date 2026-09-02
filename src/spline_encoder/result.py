"""Continuous-first result objects returned by SplineEncoder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SplineParameters:
    """One observation-aligned local action spline.

    ``control_points`` and ``knots`` always contain the continuous
    representation.  ``tokens`` is populated only when quantization was
    explicitly requested.  Knots are expressed in seconds; ``knot_steps``
    exposes their action-timestep equivalent.
    """

    control_points: FloatArray
    knots: FloatArray
    degree: int
    sample_period: float
    executable_steps: int
    tokenizer_id: str
    observation_index: int | None = None
    duration_steps: IntArray | None = None
    tokens: IntArray | None = None
    dequantized_control_points: FloatArray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        controls = np.asarray(self.control_points, dtype=np.float64)
        knots = np.asarray(self.knots, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[0] == 0 or controls.shape[1] == 0:
            raise ValueError("control_points must have shape (num_basis, action_dim)")
        if knots.ndim != 1 or len(knots) != len(controls) + self.degree + 1:
            raise ValueError("knots must have num_basis + degree + 1 elements")
        if np.any(np.diff(knots) < 0) or not np.all(np.isfinite(knots)):
            raise ValueError("knots must be finite and nondecreasing")
        if not np.all(np.isfinite(controls)):
            raise ValueError("control_points must be finite")
        if self.sample_period <= 0 or self.executable_steps < 1:
            raise ValueError("sample_period and executable_steps must be positive")
        if self.tokens is not None and np.asarray(self.tokens).shape != controls.shape:
            raise ValueError("tokens must match control_points shape")
        if (self.dequantized_control_points is not None and
                np.asarray(self.dequantized_control_points).shape != controls.shape):
            raise ValueError("dequantized_control_points must match controls")
        object.__setattr__(self, "control_points", controls.copy())
        object.__setattr__(self, "knots", knots.copy())
        if self.duration_steps is not None:
            durations = np.asarray(self.duration_steps)
            if durations.ndim != 1 or not np.issubdtype(durations.dtype, np.integer):
                raise ValueError("duration_steps must be a one-dimensional integer array")
            if np.any(durations < 1):
                raise ValueError("duration_steps must be positive")
            object.__setattr__(self, "duration_steps", durations.astype(np.int64, copy=True))
        if self.tokens is not None:
            object.__setattr__(self, "tokens", np.asarray(self.tokens, dtype=np.int64).copy())
        if self.dequantized_control_points is not None:
            object.__setattr__(
                self,
                "dequantized_control_points",
                np.asarray(self.dequantized_control_points, dtype=np.float64).copy(),
            )

    @property
    def action_dim(self) -> int:
        return int(self.control_points.shape[1])

    @property
    def num_basis(self) -> int:
        return int(self.control_points.shape[0])

    @property
    def knot_steps(self) -> FloatArray:
        return self.knots / self.sample_period

    def scipy_spline(self, *, dequantized: bool = False) -> BSpline:
        """Build a SciPy spline without extrapolation."""

        controls = self.control_points
        if dequantized:
            if self.dequantized_control_points is None:
                raise ValueError("this result has no quantized representation")
            controls = self.dequantized_control_points
        return BSpline(
            self.knots,
            controls,
            self.degree,
            extrapolate=False,
            axis=0,
        )

    def decode(
        self,
        *,
        samples: int | None = None,
        dequantized: bool = False,
    ) -> FloatArray:
        """Sample the executable curve at endpoint-exclusive native times."""

        count = self.executable_steps if samples is None else int(samples)
        if count < 1:
            raise ValueError("samples must be positive")
        if self.metadata.get("inclusive_domain", False):
            end = (self.executable_steps - 1) * self.sample_period
            times = np.linspace(0.0, end, count, endpoint=True)
        else:
            end = self.executable_steps * self.sample_period
            times = np.linspace(0.0, end, count, endpoint=False)
        return np.asarray(self.scipy_spline(dequantized=dequantized)(times), dtype=np.float64)


@dataclass(frozen=True)
class SplineGeometry:
    duration_steps: IntArray
    distinct_steps: IntArray
    knot_steps: IntArray
    greville_steps: FloatArray
    executable_spans: int
    right_context_spans: int
    executable_end_step: int
    support_end_step: int


__all__ = ["SplineGeometry", "SplineParameters"]
