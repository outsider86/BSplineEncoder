"""Raw Piper recording primitives: field extraction, clocks, and pose deltas.

Ported from ``scripts/legacy_piper/convert_piper_to_lerobot.py``,
``convert_piper_to_lerobot_delta_ee.py`` and ``preprocess_piper.py`` with the
same numerical conventions.  Only NumPy/SciPy are required.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np


JOINT_FIELDS = (("arm_joint_position", 6), ("gripper_pos", 1))
EE_FIELDS = (("arm_pos", 3), ("arm_quat", 4), ("gripper_pos", 1))


def load_payload(episode_dir: Path) -> dict[str, Any]:
    """Read ``data.pkl``.  Raw files are opened read-only and never modified."""

    path = Path(episode_dir) / "data.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a mapping payload")
    for key in ("timestamps", "observations"):
        if key not in payload:
            raise ValueError(f"{path}: missing key {key!r}")
    return dict(payload)


def validate_timestamps(values: Any, label: str, expected: int) -> np.ndarray:
    timestamps = np.asarray(values, dtype=np.float64).reshape(-1)
    if timestamps.shape != (expected,):
        raise ValueError(f"{label}: expected {expected} timestamps, got {timestamps.shape}")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise ValueError(f"{label}: timestamps must be finite and non-decreasing")
    return timestamps


def nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return source indices nearest to each target; ties choose the earlier sample."""

    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(choose_right, right, left).astype(np.int64)


def rows(records: Sequence[Any], fields: tuple[tuple[str, int], ...], label: str) -> np.ndarray:
    """Stack fixed-size float32 vectors from per-sample raw dictionaries."""

    vectors = []
    for index, row in enumerate(records):
        parts = []
        for key, dim in fields:
            if key not in row:
                raise ValueError(f"{label}[{index}]: missing {key!r}")
            value = np.asarray(row[key], dtype=np.float32).reshape(-1)
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"{label}[{index}].{key}: expected finite ({dim},), got {value}")
            parts.append(value)
        vectors.append(np.concatenate(parts))
    return np.stack(vectors)


def field_rows(records: Sequence[Any], field: str, label: str) -> np.ndarray:
    """Stack one raw field of any consistent size (used for extra observations)."""

    values = []
    for index, row in enumerate(records):
        if field not in row:
            raise ValueError(f"{label}[{index}]: missing {field!r}")
        value = np.asarray(row[field], dtype=np.float32).reshape(-1)
        if not value.size or not np.isfinite(value).all():
            raise ValueError(f"{label}[{index}].{field}: expected a finite nonempty vector")
        if values and value.shape != values[0].shape:
            raise ValueError(
                f"{label}[{index}].{field}: size {value.size} differs from {values[0].size}"
            )
        values.append(value)
    return np.stack(values)


def discover_episodes(raw_task: Path) -> list[Path]:
    """Episode directories containing ``data.pkl`` (legacy and dated layouts)."""

    return sorted({path.parent for path in Path(raw_task).rglob("data.pkl")})


__all__ = [
    "EE_FIELDS", "JOINT_FIELDS", "discover_episodes", "field_rows", "load_payload",
    "nearest_indices", "rows", "validate_timestamps",
]
