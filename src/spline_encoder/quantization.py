"""Optional calibrated scalar quantization for spline controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ControlQuantizer:
    """Independent min/max scalar quantization along configured axes."""

    vocab_size: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self.low is not None and self.high is not None

    def set_bounds(self, low: Any, high: Any) -> None:
        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        if low_array.shape != high_array.shape or np.any(low_array > high_array):
            raise ValueError("invalid quantization bounds")
        if not np.all(np.isfinite(low_array)) or not np.all(np.isfinite(high_array)):
            raise ValueError("quantization bounds must be finite")
        self.low = low_array.copy()
        self.high = high_array.copy()

    def encode(self, values: Any) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError("quantization requires calibration")
        array = np.asarray(values, dtype=np.float64)
        span = self.high - self.low
        safe = np.where(span > 0, span, 1.0)
        scaled = np.clip((array - self.low) / safe, 0.0, 1.0)
        return np.rint(scaled * (self.vocab_size - 1)).astype(np.int64)

    def decode(self, tokens: Any) -> np.ndarray:
        if not self.calibrated:
            raise RuntimeError("dequantization requires calibration")
        values = np.asarray(tokens)
        if not np.issubdtype(values.dtype, np.integer):
            raise TypeError("tokens must use an integer dtype")
        if np.any(values < 0) or np.any(values >= self.vocab_size):
            raise ValueError("control token outside vocabulary")
        return self.low + values.astype(np.float64) / (self.vocab_size - 1) * (
            self.high - self.low
        )

    def state_dict(self) -> dict[str, list]:
        if not self.calibrated:
            raise RuntimeError("quantizer is not calibrated")
        return {"low": self.low.tolist(), "high": self.high.tolist()}


__all__ = ["ControlQuantizer"]
