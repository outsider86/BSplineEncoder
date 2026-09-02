"""Typed configuration for the supported action-spline geometries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping

import numpy as np


@dataclass(frozen=True)
class UniformBSplineConfig:
    """Uniform B-spline clamped at both ends of the action chunk.

    ``span_length_steps=None`` uses evenly spaced knots over the inclusive
    first/last sample range.  Setting it uses endpoint-exclusive robot-policy
    semantics: an ``H``-action chunk lives on ``[0, H]`` and is sampled at
    integer steps ``0 .. H-1``.
    """

    action_dim: int
    chunk_size: int = 16
    frequency_hz: float = 20.0
    degree: int = 3
    num_basis: int = 8
    span_length_steps: int | None = None
    vocab_size: int = 256
    regularization: float = 1e-4
    end_padding: bool = True
    mode: Literal["uniform"] = "uniform"

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        _validate_span_layout(self)

    @property
    def num_spans(self) -> int:
        return self.num_basis - self.degree

    @property
    def input_chunk_size(self) -> int:
        return self.chunk_size

    @property
    def delta_t(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class UniformDoubleBSplineConfig(UniformBSplineConfig):
    """Canonical name for a double-clamped uniform B-spline."""

    mode: Literal["uniform_double"] = "uniform_double"


@dataclass(frozen=True)
class UniformLeftBSplineConfig(UniformBSplineConfig):
    """Left-clamped spline obtained from a longer double-clamped fit.

    The first ``num_basis - degree`` spans are executable.  The final
    ``degree`` retained controls provide right-side basis support.  The
    fitting-only control suffix is discarded after fitting.
    """

    implementation_version: str = "uniform_left_extended_double_fit_v1"
    mode: Literal["uniform_left"] = "uniform_left"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.span_length_steps is None:
            raise ValueError("uniform_left requires span_length_steps")
        if not self.implementation_version:
            raise ValueError("implementation_version must be nonempty")

    @property
    def executable_spans(self) -> int:
        return self.num_basis - self.degree

    @property
    def right_context_spans(self) -> int:
        return self.degree

    @property
    def fitting_num_spans(self) -> int:
        if self.implementation_version == "uniform_left_extended_double_fit_v1":
            return self.executable_spans + self.right_context_spans
        # Preserve loadability of the historical experimental v2 artifact.
        return self.num_basis + 1

    @property
    def fitting_num_basis(self) -> int:
        return self.fitting_num_spans + self.degree

    @property
    def fitting_chunk_size(self) -> int:
        return self.fitting_num_spans * int(self.span_length_steps)

    @property
    def input_chunk_size(self) -> int:
        return self.fitting_chunk_size

    @property
    def support_end_step(self) -> int:
        return self.num_basis * int(self.span_length_steps)


@dataclass(frozen=True)
class AdaptiveBSplineConfig:
    """Adaptive FITPACK spline with left-clamped local prefixes."""

    action_dim: int
    chunk_size: int = 16
    frequency_hz: float = 20.0
    degree: int = 3
    num_basis: int = 8
    span_length_steps: int | None = None
    vocab_size: int = 256
    fit_tolerance: float = 0.01
    smoothing: float = 1e-12
    max_duration: int = 255
    alignment_tolerance: float = 1e-8
    limit_global_knot_density: bool = True
    end_padding: bool = True
    implementation_version: str = "fitpack_left_clamped_right_open_tail_refit_v5"
    mode: Literal["adaptive"] = "adaptive"

    def __post_init__(self) -> None:
        _validate_common(self)
        if self.num_basis <= 2 * self.degree:
            raise ValueError("adaptive mode requires num_basis > 2 * degree")
        if self.fit_tolerance <= 0 or self.smoothing < 0:
            raise ValueError("invalid adaptive fit configuration")
        if self.max_duration < 1 or self.alignment_tolerance < 0:
            raise ValueError("invalid adaptive duration configuration")
        if not self.implementation_version:
            raise ValueError("implementation_version must be nonempty")
        _validate_span_layout(self)

    @property
    def delta_t(self) -> float:
        return 1.0 / self.frequency_hz

    @property
    def input_chunk_size(self) -> int:
        # Standalone chunk encoding treats one input chunk as a short episode.
        return self.chunk_size

    @property
    def average_span_steps(self) -> int:
        if self.span_length_steps is not None:
            return self.span_length_steps
        value = (self.chunk_size - 1) / (self.num_basis - self.degree)
        rounded = int(round(value))
        if not np.isclose(value, rounded, rtol=0.0, atol=1e-12):
            raise ValueError("adaptive tail padding needs integer average span steps")
        return rounded

    @property
    def tail_padding_spans(self) -> int:
        if self.implementation_version == "fitpack_left_clamped_right_open_tail_refit_v5":
            return self.num_basis
        return self.num_basis + 1

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self)


@dataclass(frozen=True)
class AdaptiveLeftBSplineConfig(AdaptiveBSplineConfig):
    mode: Literal["adaptive_left"] = "adaptive_left"


Config = (
    UniformBSplineConfig
    | UniformDoubleBSplineConfig
    | UniformLeftBSplineConfig
    | AdaptiveBSplineConfig
    | AdaptiveLeftBSplineConfig
)

UNIFORM_DOUBLE_MODES = frozenset(("uniform", "uniform_double"))
UNIFORM_LEFT_MODES = frozenset(("uniform_left",))
UNIFORM_MODES = UNIFORM_DOUBLE_MODES | UNIFORM_LEFT_MODES
ADAPTIVE_MODES = frozenset(("adaptive", "adaptive_left"))
ENCODER_MODES = tuple(sorted(UNIFORM_MODES | ADAPTIVE_MODES))


def is_uniform_mode(mode: str) -> bool:
    return str(mode) in UNIFORM_MODES


def is_adaptive_mode(mode: str) -> bool:
    return str(mode) in ADAPTIVE_MODES


def preset(name: str, mode: str = "uniform_double") -> Config:
    """Return a validated common robotics configuration."""

    key = name.lower().replace("-", "_")
    if key == "libero":
        values = dict(action_dim=7, chunk_size=16, frequency_hz=10.0,
                      num_basis=11, span_length_steps=2)
    elif key in {"libero_chunk8", "libero_span2_chunk8"}:
        values = dict(action_dim=7, chunk_size=8, frequency_hz=10.0,
                      num_basis=7, span_length_steps=2)
    elif key in {"robotwin", "robotwin_unified"}:
        values = dict(action_dim=14, chunk_size=16, frequency_hz=30.0,
                      num_basis=11, span_length_steps=2)
    elif key in {"piper_chunk50", "piper_span2_chunk50"}:
        values = dict(action_dim=7, chunk_size=50, frequency_hz=50.0,
                      num_basis=28, span_length_steps=2)
    else:
        raise KeyError(f"unknown preset: {name}")
    classes = {
        "uniform": UniformBSplineConfig,
        "uniform_double": UniformDoubleBSplineConfig,
        "uniform_left": UniformLeftBSplineConfig,
        "adaptive": AdaptiveBSplineConfig,
        "adaptive_left": AdaptiveLeftBSplineConfig,
    }
    try:
        return classes[mode](**values)
    except KeyError as exc:
        raise ValueError(f"mode must be one of {ENCODER_MODES}, got {mode!r}") from exc


benchmark_preset = preset


def config_from_dict(values: Mapping[str, Any]) -> Config:
    payload = dict(values)
    classes = {
        "uniform": UniformBSplineConfig,
        "uniform_double": UniformDoubleBSplineConfig,
        "uniform_left": UniformLeftBSplineConfig,
        "adaptive": AdaptiveBSplineConfig,
        "adaptive_left": AdaptiveLeftBSplineConfig,
    }
    mode = payload.get("mode")
    if mode not in classes:
        raise ValueError(f"unsupported encoder mode: {mode!r}")
    return classes[mode](**payload)


def _validate_common(config: Any) -> None:
    if config.action_dim < 1 or config.chunk_size < 2 or config.frequency_hz <= 0:
        raise ValueError("action_dim, chunk_size, and frequency_hz must be positive")
    if config.degree < 1 or config.degree > 5:
        raise ValueError("degree must be between 1 and 5")
    if config.num_basis < config.degree + 1:
        raise ValueError("num_basis must be at least degree + 1")
    if config.vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")


def _validate_span_layout(config: Any) -> None:
    span = config.span_length_steps
    if span is None:
        return
    if span < 1:
        raise ValueError("span_length_steps must be positive")
    if (config.num_basis - config.degree) * span != config.chunk_size:
        raise ValueError(
            "(num_basis - degree) * span_length_steps must equal chunk_size"
        )


def _fingerprint(config: Any) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()[:16]


__all__ = [
    "ADAPTIVE_MODES", "ENCODER_MODES", "UNIFORM_MODES", "AdaptiveBSplineConfig",
    "AdaptiveLeftBSplineConfig", "Config", "UniformBSplineConfig",
    "UniformDoubleBSplineConfig", "UniformLeftBSplineConfig", "benchmark_preset",
    "config_from_dict", "is_adaptive_mode", "is_uniform_mode", "preset",
]
