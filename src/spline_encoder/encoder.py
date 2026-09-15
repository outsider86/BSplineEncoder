"""Framework-independent continuous and optionally quantized spline encoders."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from .adaptive import encode_knot_durations, fit_adaptive_bspline
from .config import (
    AdaptiveBSplineConfig,
    Config,
    UniformBSplineConfig,
    UniformLeftBSplineConfig,
)
from .fitting import make_uniform_left_clamped_knots
from .quantization import ControlQuantizer
from .result import SplineGeometry, SplineParameters
from .subdivision import extract_left_clamped_prefix


def _validate_actions(actions: Any, length: int | None, dim: int) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"actions must have shape (steps, {dim}), got {values.shape}")
    if length is not None and len(values) != length:
        raise ValueError(f"actions must have shape ({length}, {dim}), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must contain only finite values")
    return values


def _design(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = len(knots) - degree - 1
    values = np.zeros((len(x), n + degree))
    for index in range(n + degree):
        values[:, index] = ((x >= knots[index]) & (x < knots[index + 1])).astype(float)
    values[x == knots[-1], n - 1] = 1.0
    for order in range(1, degree + 1):
        next_values = np.zeros((len(x), n + degree - order))
        for index in range(n + degree - order):
            left = knots[index + order] - knots[index]
            right = knots[index + order + 1] - knots[index + 1]
            if left > 0:
                next_values[:, index] += (
                    (x - knots[index]) / left * values[:, index]
                )
            if right > 0:
                next_values[:, index] += (
                    (knots[index + order + 1] - x) / right * values[:, index + 1]
                )
        values = next_values
    return values[:, :n]


def _inclusive_basis(num_basis: int, degree: int, count: int):
    internal = np.linspace(0.0, count - 1, num_basis - degree + 1)
    knots = np.r_[np.zeros(degree), internal, np.full(degree, count - 1)]
    basis = _design(np.linspace(0, count - 1, count), knots, degree)
    return basis, knots


def _fixed_span_bases(num_basis: int, degree: int, count: int, span_steps: int):
    end = (num_basis - degree) * span_steps
    breaks = np.arange(num_basis - degree + 1, dtype=np.float64) * span_steps
    knots = np.r_[np.zeros(degree), breaks, np.full(degree, end)]
    return (
        _design(np.arange(count, dtype=np.float64), knots, degree),
        _design(np.arange(count + 1, dtype=np.float64), knots, degree),
        knots,
    )


class BaseSplineEncoder:
    """Shared public surface for all spline geometries."""

    config: Config

    def __init__(self) -> None:
        self.quantizer = ControlQuantizer(self.config.vocab_size)

    @property
    def tokenizer_id(self) -> str:
        return self.config.fingerprint

    @property
    def calibrated(self) -> bool:
        return self.quantizer.calibrated

    def calibration_state(self) -> dict[str, list]:
        return self.quantizer.state_dict()

    def load_calibration(self, state: dict[str, Any]) -> None:
        self.quantizer.set_bounds(state["low"], state["high"])

    def _quantized_fields(self, controls: np.ndarray, quantize: bool):
        if not quantize:
            return None, None
        tokens = self.quantizer.encode(controls)
        return tokens, self.quantizer.decode(tokens)

    def encode(self, actions: Any, *, quantize: bool = False) -> SplineParameters:
        """Alias for :meth:`encode_chunk`."""

        return self.encode_chunk(actions, quantize=quantize)


class UniformBSplineEncoder(BaseSplineEncoder):
    """Double-clamped uniform action-chunk encoder."""

    def __init__(self, config: UniformBSplineConfig, calibration: dict | None = None):
        self.config = config
        if config.span_length_steps is None:
            self.basis, self.knots = _inclusive_basis(
                config.num_basis, config.degree, config.chunk_size
            )
            self._fit_basis = self.basis
        else:
            self.basis, self._fit_basis, self.knots = _fixed_span_bases(
                config.num_basis, config.degree, config.chunk_size,
                config.span_length_steps,
            )
        self._solver = (
            np.linalg.inv(
                self._fit_basis.T @ self._fit_basis
                + config.regularization * np.eye(config.num_basis)
            )
            @ self._fit_basis.T
        )
        super().__init__()
        if calibration is not None:
            self.load_calibration(calibration)

    def fit_control_points(self, actions: Any) -> np.ndarray:
        values = _validate_actions(actions, self.config.chunk_size, self.config.action_dim)
        if self.config.span_length_steps is not None and self.config.end_padding:
            values = np.concatenate((values, values[-1:]), axis=0)
        elif len(values) != self._fit_basis.shape[0]:
            raise ValueError("fixed-span fitting requires end_padding=True")
        return self._solver @ values

    def calibrate(self, chunks: Iterable[Any]) -> None:
        low = high = None
        count = 0
        for chunk in chunks:
            controls = self.fit_control_points(chunk)
            low = controls.copy() if low is None else np.minimum(low, controls)
            high = controls.copy() if high is None else np.maximum(high, controls)
            count += 1
        if count == 0:
            raise ValueError("cannot calibrate from an empty chunk iterable")
        self.quantizer.set_bounds(low, high)

    def encode_chunk(self, actions: Any, *, quantize: bool = False) -> SplineParameters:
        controls = self.fit_control_points(actions)
        tokens, dequantized = self._quantized_fields(controls, quantize)
        return SplineParameters(
            control_points=controls,
            knots=self.knots * self.config.delta_t,
            degree=self.config.degree,
            sample_period=self.config.delta_t,
            executable_steps=self.config.chunk_size,
            tokenizer_id=self.tokenizer_id,
            tokens=tokens,
            dequantized_control_points=dequantized,
            metadata={
                "mode": self.config.mode,
                "boundary_condition": "double_clamped",
                "inclusive_domain": self.config.span_length_steps is None,
            },
        )

    def decode_control_points(self, tokens: Any) -> np.ndarray:
        values = np.asarray(tokens)
        expected = (self.config.num_basis, self.config.action_dim)
        if values.shape != expected:
            raise ValueError(f"tokens must have shape {expected}")
        return self.quantizer.decode(values)

    def decode(self, tokens: Any) -> np.ndarray:
        return self.basis @ self.decode_control_points(tokens)

    def load_calibration(self, state: dict[str, Any]) -> None:
        super().load_calibration(state)
        expected = (self.config.num_basis, self.config.action_dim)
        if self.quantizer.low.shape != expected:
            raise ValueError(f"uniform calibration must have shape {expected}")


class UniformLeftBSplineEncoder(UniformBSplineEncoder):
    """Fit a left-clamped/right-open spline directly on one action chunk."""

    def __init__(self, config: UniformLeftBSplineConfig, calibration: dict | None = None):
        self.config = config
        self.knots = make_uniform_left_clamped_knots(
            0.0,
            float(config.chunk_size),
            num_basis=config.num_basis,
            degree=config.degree,
        )
        fit_steps = config.chunk_size + int(config.end_padding)
        self._fit_basis = _design(
            np.arange(fit_steps, dtype=np.float64),
            self.knots,
            config.degree,
        )
        self.basis = _design(
            np.arange(config.chunk_size, dtype=np.float64), self.knots, config.degree
        )
        regularized = (
            self._fit_basis.T @ self._fit_basis
            + config.regularization * np.eye(config.num_basis)
        )
        self._solver = np.linalg.solve(regularized, self._fit_basis.T)
        BaseSplineEncoder.__init__(self)
        if calibration is not None:
            self.load_calibration(calibration)

    def fit_control_points(self, actions: Any) -> np.ndarray:
        values = _validate_actions(
            actions, self.config.chunk_size, self.config.action_dim
        )
        if self.config.end_padding:
            values = np.concatenate((values, values[-1:]), axis=0)
        return self._solver @ values

    def geometry(self) -> SplineGeometry:
        durations = np.full(
            self.config.num_basis, int(self.config.span_length_steps), dtype=np.int64
        )
        distinct = np.r_[0, np.cumsum(durations)].astype(np.int64)
        knots = np.rint(self.knots).astype(np.int64)
        greville = np.asarray([
            knots[index + 1 : index + self.config.degree + 1].mean()
            for index in range(self.config.num_basis)
        ])
        return SplineGeometry(
            duration_steps=durations,
            distinct_steps=distinct,
            knot_steps=knots,
            greville_steps=greville,
            executable_spans=self.config.executable_spans,
            right_context_spans=self.config.right_context_spans,
            executable_end_step=self.config.chunk_size,
            support_end_step=self.config.support_end_step,
        )

    def encode_chunk(self, actions: Any, *, quantize: bool = False) -> SplineParameters:
        controls = self.fit_control_points(actions)
        tokens, dequantized = self._quantized_fields(controls, quantize)
        geometry = self.geometry()
        return SplineParameters(
            control_points=controls,
            knots=self.knots * self.config.delta_t,
            degree=self.config.degree,
            sample_period=self.config.delta_t,
            executable_steps=self.config.chunk_size,
            tokenizer_id=self.tokenizer_id,
            duration_steps=geometry.duration_steps,
            tokens=tokens,
            dequantized_control_points=dequantized,
            metadata={"mode": self.config.mode,
                      "boundary_condition": "left_clamped_right_open",
                      "implementation_version": self.config.implementation_version},
        )

    def decode_spline(self, tokens: Any) -> BSpline:
        return BSpline(
            self.knots * self.config.delta_t,
            self.decode_control_points(tokens),
            self.config.degree,
            extrapolate=False,
            axis=0,
        )

    def decode(self, tokens: Any) -> np.ndarray:
        times = np.arange(self.config.chunk_size) * self.config.delta_t
        return np.asarray(self.decode_spline(tokens)(times), dtype=np.float64)


class _LegacyExtendedUniformLeftBSplineEncoder(UniformLeftBSplineEncoder):
    """Compatibility loader for the superseded extended-double-fit artifacts."""

    def __init__(self, config: UniformLeftBSplineConfig, calibration: dict | None = None):
        self.config = config
        span = int(config.span_length_steps)
        support_end = config.fitting_num_spans * span
        breaks = np.arange(config.fitting_num_spans + 1, dtype=np.float64) * span
        self.full_knots = np.concatenate((
            np.zeros(config.degree),
            breaks,
            np.full(config.degree, support_end),
        ))
        self._fit_basis = _design(
            np.arange(config.fitting_chunk_size + 1, dtype=np.float64),
            self.full_knots,
            config.degree,
        )
        self.full_basis = _design(
            np.arange(config.fitting_chunk_size, dtype=np.float64),
            self.full_knots,
            config.degree,
        )
        self.knots = self.full_knots[: config.num_basis + config.degree + 1].copy()
        self.basis = _design(
            np.arange(config.chunk_size, dtype=np.float64), self.knots, config.degree
        )
        regularized = (
            self._fit_basis.T @ self._fit_basis
            + config.regularization * np.eye(config.fitting_num_basis)
        )
        self._solver = np.linalg.solve(regularized, self._fit_basis.T)
        self._retained_control_indices = np.arange(config.num_basis, dtype=np.int64)
        BaseSplineEncoder.__init__(self)
        if calibration is not None:
            self.load_calibration(calibration)

    @property
    def retained_control_indices(self) -> np.ndarray:
        return self._retained_control_indices.copy()

    def fit_full_control_points(self, actions: Any) -> np.ndarray:
        values = _validate_actions(
            actions, self.config.fitting_chunk_size, self.config.action_dim
        )
        if not self.config.end_padding:
            raise ValueError("legacy uniform_left fitting requires end_padding=True")
        return self._solver @ np.concatenate((values, values[-1:]), axis=0)

    def fit_control_points(self, actions: Any) -> np.ndarray:
        return self.fit_full_control_points(actions)[self._retained_control_indices]


class AdaptiveBSplineEncoder(BaseSplineEncoder):
    """Adaptive episode encoder with exact observation-aligned subdivision."""

    def __init__(self, config: AdaptiveBSplineConfig, calibration: dict | None = None):
        self.config = config
        super().__init__()
        if calibration is not None:
            self.load_calibration(calibration)

    def max_episode_control_points(self, num_steps: int) -> int | None:
        if not self.config.limit_global_knot_density:
            return None
        if num_steps <= self.config.degree:
            raise ValueError("episode is too short for the configured degree")
        spans = int(np.floor(
            (num_steps - 1) / self.config.average_span_steps
            + self.config.alignment_tolerance
        ))
        return max(self.config.num_basis, spans + self.config.degree)

    def _refit_with_repeated_tail(self, spline, actions, timestamps):
        if not self.config.end_padding:
            return spline
        degree = int(spline.k)
        knots = np.asarray(spline.t, dtype=np.float64)
        end = float(knots[-degree - 1])
        padding_steps = self.config.tail_padding_spans * self.config.average_span_steps
        final = end + padding_steps * self.config.delta_t
        extended_times = np.concatenate((
            timestamps,
            end + np.arange(1, padding_steps + 1) * self.config.delta_t,
        ))
        extended_actions = np.concatenate((
            actions,
            np.repeat(actions[-1:], padding_steps, axis=0),
        ))
        extended_knots = np.concatenate((
            knots[: -(degree + 1)],
            end + np.arange(self.config.tail_padding_spans)
            * self.config.average_span_steps * self.config.delta_t,
            np.repeat(final, degree + 1),
        ))
        subdivisions = degree + 1
        dense_count = (len(actions) - 1 + padding_steps) * subdivisions + 1
        dense_times = np.linspace(float(timestamps[0]), final, dense_count)
        dense_actions = np.empty((dense_count, actions.shape[1]), dtype=np.float64)
        original = dense_times <= end
        dense_actions[original] = spline(dense_times[original])
        dense_actions[~original] = actions[-1]
        dense_actions[np.arange(len(actions)) * subdivisions] = actions
        extended = make_lsq_spline(
            dense_times, dense_actions, extended_knots, k=degree, axis=0
        )
        residual = np.asarray(extended(extended_times) - extended_actions)
        if not np.all(np.isfinite(extended.c)) or not np.all(np.isfinite(residual)):
            raise RuntimeError("adaptive repeated-action tail refit is non-finite")
        return extended

    def episode_control_points(self, actions: Any):
        values = _validate_actions(actions, None, self.config.action_dim)
        if len(values) <= self.config.degree:
            return []
        timestamps = np.arange(len(values), dtype=np.float64) * self.config.delta_t
        fit = fit_adaptive_bspline(
            values,
            timestamps=timestamps,
            degree=self.config.degree,
            max_error=self.config.fit_tolerance,
            smoothing=self.config.smoothing,
            max_control_points=self.max_episode_control_points(len(values)),
            knot_insertion_excluded_dimensions=(
                self.config.knot_insertion_excluded_dimensions
            ),
        )
        spline = self._refit_with_repeated_tail(fit.spline, values, timestamps)
        breaks = np.unique(spline.t)
        records = []
        for observation, timestamp in enumerate(timestamps):
            left = np.searchsorted(breaks, timestamp, side="right") - 1
            if left < 0 or left + self.config.num_basis >= len(breaks):
                continue
            prefix = extract_left_clamped_prefix(
                spline, float(timestamp), num_knot_spans=self.config.num_basis
            )
            durations = encode_knot_durations(
                prefix.distinct_knots,
                delta_t=self.config.delta_t,
                tolerance=self.config.alignment_tolerance,
                max_duration=self.config.max_duration,
            )
            if len(prefix.control_points) != self.config.num_basis:
                raise RuntimeError("adaptive prefix shape mismatch")
            records.append((observation, prefix.control_points.copy(), durations))
        return records

    def calibrate(self, episodes: Iterable[Any]) -> None:
        low = high = None
        count = 0
        for episode in episodes:
            for _, controls, _ in self.episode_control_points(episode):
                local_low, local_high = controls.min(axis=0), controls.max(axis=0)
                low = local_low.copy() if low is None else np.minimum(low, local_low)
                high = local_high.copy() if high is None else np.maximum(high, local_high)
                count += 1
        if count == 0:
            raise ValueError("no valid adaptive records to calibrate")
        self.quantizer.set_bounds(low, high)

    def geometry(self, duration_steps: Any) -> SplineGeometry:
        durations = np.asarray(duration_steps)
        if durations.shape != (self.config.num_basis,) or not np.issubdtype(
            durations.dtype, np.integer
        ) or np.any(durations < 1):
            raise ValueError("duration_steps must be positive integers matching num_basis")
        durations = durations.astype(np.int64, copy=False)
        distinct = np.r_[0, np.cumsum(durations)].astype(np.int64)
        knots = np.r_[np.zeros(self.config.degree + 1, dtype=np.int64), distinct[1:]]
        greville = np.asarray([
            knots[index + 1 : index + self.config.degree + 1].mean()
            for index in range(self.config.num_basis)
        ])
        executable_spans = self.config.num_basis - self.config.degree
        return SplineGeometry(
            duration_steps=durations.copy(),
            distinct_steps=distinct,
            knot_steps=knots,
            greville_steps=greville,
            executable_spans=executable_spans,
            right_context_spans=self.config.degree,
            executable_end_step=int(distinct[executable_spans]),
            support_end_step=int(distinct[-1]),
        )

    def _make_result(self, observation: int, controls: np.ndarray,
                     durations: np.ndarray, quantize: bool) -> SplineParameters:
        geometry = self.geometry(durations)
        tokens, dequantized = self._quantized_fields(controls, quantize)
        return SplineParameters(
            control_points=controls,
            knots=geometry.knot_steps.astype(np.float64) * self.config.delta_t,
            degree=self.config.degree,
            sample_period=self.config.delta_t,
            executable_steps=geometry.executable_end_step,
            tokenizer_id=self.tokenizer_id,
            observation_index=observation,
            duration_steps=durations,
            tokens=tokens,
            dequantized_control_points=dequantized,
            metadata={"mode": self.config.mode,
                      "boundary_condition": "left_clamped_right_open",
                      "policy_layout": "control_point_duration_interleaved_v1",
                      "knot_insertion_excluded_dimensions": list(
                          self.config.knot_insertion_excluded_dimensions
                      )},
        )

    def encode_episode(self, actions: Any, *, quantize: bool = False):
        return [
            self._make_result(observation, controls, durations, quantize)
            for observation, controls, durations in self.episode_control_points(actions)
        ]

    def encode_chunk(self, actions: Any, *, quantize: bool = False) -> SplineParameters:
        values = _validate_actions(
            actions, self.config.input_chunk_size, self.config.action_dim
        )
        records = self.encode_episode(values, quantize=quantize)
        if not records or records[0].observation_index != 0:
            raise RuntimeError("adaptive chunk did not produce an observation-zero prefix")
        return records[0]

    def decode_control_points(self, tokens: Any) -> np.ndarray:
        values = np.asarray(tokens)
        expected = (self.config.num_basis, self.config.action_dim)
        if values.shape != expected:
            raise ValueError(f"tokens must have shape {expected}")
        return self.quantizer.decode(values)

    def decode_spline(self, tokens: Any, duration_steps: Any) -> BSpline:
        geometry = self.geometry(duration_steps)
        return BSpline(
            geometry.knot_steps.astype(np.float64) * self.config.delta_t,
            self.decode_control_points(tokens),
            self.config.degree,
            extrapolate=False,
            axis=0,
        )

    def decode(self, tokens: Any, duration_steps: Any,
               samples: int | None = None) -> np.ndarray:
        geometry = self.geometry(duration_steps)
        count = geometry.executable_end_step if samples is None else int(samples)
        if count < 1:
            raise ValueError("samples must be positive")
        steps = np.linspace(0.0, geometry.executable_end_step, count, endpoint=False)
        return np.asarray(
            self.decode_spline(tokens, duration_steps)(steps * self.config.delta_t),
            dtype=np.float64,
        )

    def load_calibration(self, state: dict[str, Any]) -> None:
        super().load_calibration(state)
        if self.quantizer.low.shape != (self.config.action_dim,):
            raise ValueError("adaptive calibration must be per action dimension")


UniformDoubleBSplineEncoder = UniformBSplineEncoder
AdaptiveLeftBSplineEncoder = AdaptiveBSplineEncoder


def create_encoder(config: Config, calibration: dict | None = None) -> BaseSplineEncoder:
    if isinstance(config, UniformLeftBSplineConfig):
        if not config.uses_direct_fit:
            return _LegacyExtendedUniformLeftBSplineEncoder(config, calibration)
        return UniformLeftBSplineEncoder(config, calibration)
    if isinstance(config, UniformBSplineConfig):
        return UniformBSplineEncoder(config, calibration)
    if isinstance(config, AdaptiveBSplineConfig):
        return AdaptiveBSplineEncoder(config, calibration)
    raise TypeError(f"unsupported configuration: {type(config).__name__}")


def encode_action_chunk(
    actions: Any,
    config: Config,
    *,
    quantize: bool = False,
    calibration: dict | None = None,
) -> SplineParameters:
    """One-call action-chunk encoding convenience API."""

    return create_encoder(config, calibration).encode_chunk(actions, quantize=quantize)


__all__ = [
    "AdaptiveBSplineEncoder", "AdaptiveLeftBSplineEncoder", "BaseSplineEncoder",
    "UniformBSplineEncoder", "UniformDoubleBSplineEncoder",
    "UniformLeftBSplineEncoder", "create_encoder", "encode_action_chunk",
]
