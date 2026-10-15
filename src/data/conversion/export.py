"""LeRobot v2.1 writer for aligned Piper episodes (PyArrow imported lazily).

Layout (``chunks_size`` 1000, chunk = episode_index // 1000)::

    data/chunk-XXX/episode_XXXXXX.parquet
    videos/chunk-XXX/<video key>/episode_XXXXXX.mp4
    meta/info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl
    meta/modality.json, embodiment.json                (GR00T/starVLA metadata)
    meta/cleanup_summary.json, cleanup_metrics.csv     (idle cleanup)
    meta/source_episodes.jsonl                          (raw episode mapping)
    meta/source_alignment/chunk-XXX/episode_XXXXXX.parquet
        per-row raw robot index, raw-clock target time, camera source frame
        indices, and alignment errors.  Kept out of ``data/`` so LeRobot
        readers see exactly the declared features.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .episode import Episode


CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
ALIGNMENT_PATH = "meta/source_alignment/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("LeRobot export needs `pip install spline-encoder[piper]`") from exc
    return pa, pq


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode_path(template: str, index: int, video_key: str | None = None) -> str:
    return template.format(episode_chunk=index // CHUNKS_SIZE, episode_index=index,
                           video_key=video_key)


def numeric_stats(values: np.ndarray) -> dict[str, list]:
    """LeRobot v2.1 per-episode statistics (population std, ``count`` list)."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return {"min": array.min(0).tolist(), "max": array.max(0).tolist(),
            "mean": array.mean(0).tolist(), "std": array.std(0).tolist(),
            "count": [int(len(array))]}


def frame_columns(episode: Episode, episode_index: int, global_offset: int,
                  state: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    count = episode.length
    columns: dict[str, np.ndarray] = {"observation.state": state}
    columns.update(episode.extras)
    columns.update({
        "action": action,
        "timestamp": episode.timestamps.astype(np.float32),
        "frame_index": np.arange(count, dtype=np.int64),
        "episode_index": np.full(count, episode_index, dtype=np.int64),
        "index": np.arange(global_offset, global_offset + count, dtype=np.int64),
        "task_index": np.zeros(count, dtype=np.int64),
    })
    return columns


def write_episode_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    pa, pq = _pyarrow()
    arrays = {}
    for name, values in columns.items():
        values = np.asarray(values)
        if values.ndim == 2:
            arrays[name] = pa.array(values.astype(np.float32).tolist(),
                                    type=pa.list_(pa.float32(), values.shape[1]))
        elif values.dtype == np.float32:
            arrays[name] = pa.array(values, type=pa.float32())
        else:
            arrays[name] = pa.array(values.astype(np.int64), type=pa.int64())
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def write_alignment_parquet(path: Path, episode: Episode) -> None:
    pa, pq = _pyarrow()
    arrays = {
        "frame_index": pa.array(np.arange(episode.length, dtype=np.int64)),
        "source_robot_index": pa.array(episode.robot_indices.astype(np.int64)),
        "source_time": pa.array(episode.source_time.astype(np.float64)),
        "robot_alignment_ms": pa.array(episode.alignment_ms["robot"].astype(np.float64)),
    }
    for name, indices in episode.camera_indices.items():
        arrays[f"camera.{name}.source_frame_index"] = pa.array(indices.astype(np.int64))
        arrays[f"camera.{name}.alignment_ms"] = pa.array(
            episode.alignment_ms[name].astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), path)


def features(cameras: dict[str, tuple[int, int]], camera_keys: dict[str, str], fps: float,
             state_names: list[str], action_names: list[str],
             extras: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (height, width) in cameras.items():
        result[camera_keys[name]] = {
            "dtype": "video", "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {"video.height": height, "video.width": width, "video.channels": 3,
                     "video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    result["observation.state"] = {"dtype": "float32", "shape": [len(state_names)],
                                   "names": state_names}
    for key, dim in extras.items():
        result[key] = {"dtype": "float32", "shape": [dim],
                       "names": [f"{key.split('.', 1)[1]}_{i}" for i in range(dim)]}
    result["action"] = {"dtype": "float32", "shape": [len(action_names)], "names": action_names}
    for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                        ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        result[name] = {"dtype": dtype, "shape": [1], "names": None}
    return result


__all__ = ["ALIGNMENT_PATH", "CHUNKS_SIZE", "DATA_PATH", "VIDEO_PATH", "episode_path",
           "features", "frame_columns", "numeric_stats", "write_alignment_parquet",
           "write_episode_parquet", "write_json", "write_jsonl", "write_metrics_csv"]
