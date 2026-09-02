"""Dependency-light LeRobot v2.1/v3 action ingestion and dataset encoding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Iterable, Iterator

import numpy as np

from .config import is_adaptive_mode
from .result import SplineParameters
from .serialization import save_encoder


@dataclass(frozen=True)
class ActionAlignment:
    """Map a LeRobot action vector into the encoder's action coordinates.

    ``action_indices`` may select and reorder dimensions.  Optional bounds
    apply min/max normalization to ``[-1, 1]`` after selection.  This makes
    the alignment explicit and serializable instead of burying robot-specific
    assumptions in the encoder.
    """

    action_key: str = "action"
    action_indices: tuple[int, ...] | None = None
    low: tuple[float, ...] | None = None
    high: tuple[float, ...] | None = None
    clip: bool = True

    def __post_init__(self) -> None:
        if not self.action_key:
            raise ValueError("action_key must be nonempty")
        if self.action_indices is not None:
            if not self.action_indices or any(index < 0 for index in self.action_indices):
                raise ValueError("action_indices must contain non-negative indices")
        if (self.low is None) != (self.high is None):
            raise ValueError("low and high must be supplied together")
        if self.low is not None:
            low, high = np.asarray(self.low), np.asarray(self.high)
            if low.ndim != 1 or low.shape != high.shape or np.any(low > high):
                raise ValueError("normalization bounds are invalid")
            if self.action_indices is not None and len(low) != len(self.action_indices):
                raise ValueError("normalization bounds must match selected dimensions")

    def apply(self, actions) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("episode actions must have shape (steps, dimensions)")
        if self.action_indices is not None:
            if max(self.action_indices) >= values.shape[1]:
                raise ValueError("an action index exceeds the dataset action width")
            values = values[:, self.action_indices]
        if self.low is not None:
            low, high = np.asarray(self.low), np.asarray(self.high)
            if len(low) != values.shape[1]:
                raise ValueError("normalization bounds do not match action width")
            span = high - low
            values = np.where(span != 0, 2 * (values - low) / span - 1, 0.0)
            if self.clip:
                values = np.clip(values, -1.0, 1.0)
        return values

    def to_dict(self) -> dict:
        return {
            "action_key": self.action_key,
            "action_indices": (None if self.action_indices is None
                               else list(self.action_indices)),
            "low": None if self.low is None else list(self.low),
            "high": None if self.high is None else list(self.high),
            "clip": self.clip,
        }


@dataclass(frozen=True)
class LeRobotEpisode:
    episode_index: int
    frame_indices: np.ndarray
    actions: np.ndarray
    timestamps: np.ndarray | None = None


@dataclass(frozen=True)
class EncodedEpisode:
    source: LeRobotEpisode
    splines: tuple[SplineParameters, ...]


@dataclass(frozen=True)
class DatasetEncodingResult:
    output_dir: Path
    manifest_path: Path
    episode_count: int
    record_count: int
    quantized: bool
    tokenizer_id: str


def _read_info(dataset: Path) -> dict:
    path = dataset / "meta/info.json"
    try:
        info = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LeRobot metadata at {path}") from exc
    if info.get("codebase_version") not in {"v2.1", "v3.0"}:
        raise ValueError("expected a standard LeRobot v2.1 or v3.0 dataset")
    return info


def load_lerobot_episodes(
    dataset: str | Path,
    alignment: ActionAlignment | None = None,
) -> tuple[list[LeRobotEpisode], dict]:
    """Load only action/index columns from a standard local LeRobot dataset."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "LeRobot loading requires `pip install spline-encoder[lerobot]`"
        ) from exc
    root = Path(dataset).expanduser().resolve()
    alignment = alignment or ActionAlignment()
    info = _read_info(root)
    feature = info.get("features", {}).get(alignment.action_key)
    if not isinstance(feature, dict) or feature.get("dtype") not in {
        "float16", "float32", "float64"
    }:
        raise ValueError(f"{alignment.action_key!r} is not a vector action feature")
    paths = sorted((root / "data").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"no parquet action files found under {root / 'data'}")
    available = set(pd.read_parquet(paths[0]).columns)
    required = {"episode_index", "frame_index", alignment.action_key}
    missing = required - available
    if missing:
        raise ValueError(f"LeRobot parquet is missing columns: {sorted(missing)}")
    columns = list(required)
    if "timestamp" in available:
        columns.append("timestamp")
    frames = [pd.read_parquet(path, columns=columns) for path in paths]
    table = pd.concat(frames, ignore_index=True).sort_values(
        ["episode_index", "frame_index"]
    )
    episodes: list[LeRobotEpisode] = []
    for episode_index, rows in table.groupby("episode_index", sort=True):
        raw = np.stack(rows[alignment.action_key].to_numpy()).astype(np.float64)
        actions = alignment.apply(raw)
        timestamps = (
            rows["timestamp"].to_numpy(dtype=np.float64)
            if "timestamp" in rows else None
        )
        episodes.append(LeRobotEpisode(
            episode_index=int(episode_index),
            frame_indices=rows["frame_index"].to_numpy(dtype=np.int64),
            actions=actions,
            timestamps=timestamps,
        ))
    return episodes, info


def infer_action_bounds(
    dataset: str | Path,
    *,
    action_key: str = "action",
    action_indices: tuple[int, ...] | None = None,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> ActionAlignment:
    """Build a min/max alignment from robust dataset-wide quantiles."""

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    base = ActionAlignment(action_key=action_key, action_indices=action_indices)
    episodes, _ = load_lerobot_episodes(dataset, base)
    actions = np.concatenate([episode.actions for episode in episodes], axis=0)
    return ActionAlignment(
        action_key=action_key,
        action_indices=action_indices,
        low=tuple(np.quantile(actions, lower_quantile, axis=0).tolist()),
        high=tuple(np.quantile(actions, upper_quantile, axis=0).tolist()),
        clip=True,
    )


def action_windows(episode: np.ndarray, count: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield every observation-aligned future-action window with tail padding."""

    if count < 1 or len(episode) < 1:
        raise ValueError("window count and episode length must be positive")
    for observation in range(len(episode)):
        window = episode[observation : observation + count]
        if len(window) < count:
            window = np.concatenate((
                window,
                np.repeat(window[-1:], count - len(window), axis=0),
            ))
        yield observation, window


def calibrate_from_lerobot(encoder, episodes: Iterable[LeRobotEpisode]) -> None:
    episodes = list(episodes)
    if is_adaptive_mode(encoder.config.mode):
        encoder.calibrate(episode.actions for episode in episodes)
    else:
        encoder.calibrate(
            window
            for episode in episodes
            for _, window in action_windows(
                episode.actions, encoder.config.input_chunk_size
            )
        )


def iter_encoded_episodes(
    encoder,
    episodes: Iterable[LeRobotEpisode],
    *,
    quantize: bool = False,
) -> Iterator[EncodedEpisode]:
    """Encode episodes while preserving a record for every observation."""

    for episode in episodes:
        if episode.actions.shape[1] != encoder.config.action_dim:
            raise ValueError(
                f"aligned action width {episode.actions.shape[1]} does not match "
                f"encoder action_dim {encoder.config.action_dim}"
            )
        if is_adaptive_mode(encoder.config.mode):
            splines = encoder.encode_episode(episode.actions, quantize=quantize)
        else:
            splines = [
                replace(
                    encoder.encode_chunk(window, quantize=quantize),
                    observation_index=observation,
                )
                for observation, window in action_windows(
                    episode.actions, encoder.config.input_chunk_size
                )
            ]
        if len(splines) != len(episode.actions):
            raise RuntimeError(
                f"episode {episode.episode_index} produced {len(splines)} records "
                f"for {len(episode.actions)} observations"
            )
        yield EncodedEpisode(episode, tuple(splines))


def encode_lerobot_dataset(
    dataset: str | Path,
    output_dir: str | Path,
    encoder,
    *,
    alignment: ActionAlignment | None = None,
    quantize: bool = False,
    calibrate_quantizer: bool = False,
    overwrite: bool = False,
) -> DatasetEncodingResult:
    """Encode a local LeRobot dataset to observation-aligned NPZ episodes.

    The output is intentionally separate from the source dataset.  Each NPZ
    contains dense ``controls`` and ``knots`` arrays indexed one-to-one by the
    source episode's ``frame_index``.  ``tokens`` is omitted unless requested.
    """

    root = Path(dataset).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    alignment = alignment or ActionAlignment()
    episodes, info = load_lerobot_episodes(root, alignment)
    if not np.isclose(float(info["fps"]), encoder.config.frequency_hz):
        raise ValueError(
            f"dataset fps {info['fps']} does not match encoder frequency_hz "
            f"{encoder.config.frequency_hz}"
        )
    if calibrate_quantizer:
        calibrate_from_lerobot(encoder, episodes)
    if quantize and not encoder.calibrated:
        raise RuntimeError("quantize=True requires calibration or calibrate_quantizer=True")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {target}")
        shutil.rmtree(target)
    (target / "data").mkdir(parents=True)
    record_count = 0
    source_digest = sha256()
    for encoded in iter_encoded_episodes(encoder, episodes, quantize=quantize):
        episode = encoded.source
        splines = encoded.splines
        source_digest.update(episode.frame_indices.astype("<i8", copy=False).tobytes())
        source_digest.update(episode.actions.astype("<f8", copy=False).tobytes())
        payload = {
            "episode_index": np.asarray(episode.episode_index, dtype=np.int64),
            "frame_index": episode.frame_indices,
            "observation_index": np.asarray(
                [item.observation_index for item in splines], dtype=np.int64
            ),
            "controls": np.stack([item.control_points for item in splines]),
            "knots": np.stack([item.knots for item in splines]),
            "executable_steps": np.asarray(
                [item.executable_steps for item in splines], dtype=np.int64
            ),
        }
        if splines[0].duration_steps is not None:
            payload["duration_steps"] = np.stack(
                [item.duration_steps for item in splines]
            )
        if quantize:
            payload["tokens"] = np.stack([item.tokens for item in splines])
            payload["dequantized_controls"] = np.stack(
                [item.dequantized_control_points for item in splines]
            )
        np.savez_compressed(
            target / "data" / f"episode_{episode.episode_index:06d}.npz", **payload
        )
        record_count += len(splines)
    save_encoder(encoder, target / "encoder.json")
    manifest = {
        "schema_version": 1,
        "format": "spline_encoder_observation_aligned_npz",
        "source_dataset": str(root),
        "source_codebase_version": info["codebase_version"],
        "source_fps": info["fps"],
        "source_action_fingerprint": source_digest.hexdigest(),
        "alignment": alignment.to_dict(),
        "tokenizer_id": encoder.tokenizer_id,
        "mode": encoder.config.mode,
        "episode_count": len(episodes),
        "record_count": record_count,
        "quantized": quantize,
        "continuous_controls": True,
        "episode_path": "data/episode_{episode_index:06d}.npz",
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return DatasetEncodingResult(
        output_dir=target,
        manifest_path=manifest_path,
        episode_count=len(episodes),
        record_count=record_count,
        quantized=quantize,
        tokenizer_id=encoder.tokenizer_id,
    )


__all__ = [
    "ActionAlignment", "DatasetEncodingResult", "EncodedEpisode", "LeRobotEpisode",
    "action_windows", "calibrate_from_lerobot", "encode_lerobot_dataset",
    "infer_action_bounds", "iter_encoded_episodes", "load_lerobot_episodes",
]
