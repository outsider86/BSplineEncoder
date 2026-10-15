#!/usr/bin/env python3
"""Convert Piper RawData to 30 Hz LeRobot data with measured delta-EE actions.

This converter is deliberately separate from ``convert_piper_to_lerobot.py``
so existing joint-state/joint-action conversion remains backward compatible.

For every retained 30 Hz sample ``t``:

* ``observation.state`` is the absolute measured TCP position, canonicalized
  measured TCP quaternion (XYZW), and absolute measured gripper at ``t``;
* ``action`` is the measured transform from TCP ``t`` to TCP ``t + 1``, encoded
  as translation in the current TCP frame and a local XYZ rotation vector,
  followed by the absolute measured gripper at ``t + 1``; and
* both videos show the current observation at ``t``.

The final resampled frame of each episode is omitted because it has no measured
next-step target. Absolute EE command/FK values in the raw ``actions`` rows are
never used as targets by this converter.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from convert_piper_to_lerobot import (
    ACTION_DIM,
    CAMERAS,
    CameraStream,
    Episode,
    _features,
    _nearest_indices,
    _validate_timestamps,
    _video_properties,
    _write_episode_parquet,
    _write_json,
    _write_resampled_video,
)


STATE_DIM = 8
STATE_SOURCE = "observation_ee"
ACTION_SOURCE = "observation_delta_ee"
STATE_LAYOUT = "measured_tcp_position[t] + measured_tcp_quaternion_xyzw[t] + measured_gripper[t]"
ACTION_LAYOUT = (
    "local_delta_tcp_translation[t->t+1] + local_delta_tcp_rotvec[t->t+1] "
    "+ measured_gripper[t+1]"
)


def _canonicalize_quaternions(quaternions: np.ndarray, label: str) -> np.ndarray:
    """Normalize XYZW quaternions and choose a continuous per-episode sign."""
    values = np.asarray(quaternions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{label}: expected quaternion shape (N, 4), got {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms < 1e-8):
        raise ValueError(f"{label}: quaternion contains NaN, Inf, or a near-zero norm")
    values /= norms[:, None]
    if values[0, 3] < 0:
        values[0] *= -1
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0:
            values[index] *= -1
    return values.astype(np.float32)


def _measured_ee_rows(data_path: Path, observations: list[Any]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for frame_index, observation in enumerate(observations):
        try:
            position = np.asarray(observation["arm_pos"], dtype=np.float32).reshape(-1)
            quaternion = np.asarray(observation["arm_quat"], dtype=np.float32).reshape(-1)
            gripper = np.asarray(observation["gripper_pos"], dtype=np.float32).reshape(-1)
        except KeyError as error:
            raise ValueError(
                f"{data_path}: observation {frame_index} is missing {error.args[0]!r}"
            ) from error
        if position.shape != (3,) or quaternion.shape != (4,) or gripper.shape != (1,):
            raise ValueError(
                f"{data_path}: observation {frame_index}: position={position.shape}, "
                f"quaternion={quaternion.shape}, gripper={gripper.shape}; expected "
                "(3,), (4,), and (1,)"
            )
        row = np.concatenate([position, quaternion, gripper])
        if not np.isfinite(row).all():
            raise ValueError(f"{data_path}: observation {frame_index} contains NaN or Inf")
        rows.append(row)
    measured = np.stack(rows).astype(np.float32, copy=False)
    measured[:, 3:7] = _canonicalize_quaternions(
        measured[:, 3:7], f"{data_path}: measured TCP"
    )
    return measured


def _next_step_actions(measured_ee: np.ndarray) -> np.ndarray:
    """Build body-frame measured delta pose and next measured gripper."""
    if measured_ee.ndim != 2 or measured_ee.shape[1] != STATE_DIM:
        raise ValueError(f"measured EE must have shape (N, {STATE_DIM}), got {measured_ee.shape}")
    if len(measured_ee) < 2:
        raise ValueError("at least two measured EE samples are required")
    rotations = Rotation.from_quat(measured_ee[:, 3:7].astype(np.float64))
    current_rotation = rotations[:-1]
    next_rotation = rotations[1:]
    world_translation = (
        measured_ee[1:, :3].astype(np.float64)
        - measured_ee[:-1, :3].astype(np.float64)
    )
    local_translation = current_rotation.inv().apply(world_translation)
    local_rotvec = (current_rotation.inv() * next_rotation).as_rotvec()
    next_gripper = measured_ee[1:, 7:8].astype(np.float64)
    action = np.concatenate(
        [local_translation, local_rotvec, next_gripper], axis=1
    ).astype(np.float32)
    if action.shape != (len(measured_ee) - 1, ACTION_DIM):
        raise RuntimeError(f"constructed action has unexpected shape {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("constructed delta-EE action contains NaN or Inf")
    return action


def _load_episode(path: Path, target_fps: float) -> Episode:
    data_path = path / "data.pkl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing {data_path}")
    with data_path.open("rb") as stream:
        payload: dict[str, Any] = pickle.load(stream)

    required = {"timestamps", "observations"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{data_path}: missing keys {sorted(missing)}")
    observations = payload["observations"]
    count = len(payload["timestamps"])
    if count == 0 or len(observations) != count:
        raise ValueError(
            f"{data_path}: lengths timestamps={count}, observations={len(observations)}"
        )
    robot_timestamps = _validate_timestamps(
        payload["timestamps"], f"{data_path}: robot", count
    )
    measured_ee = _measured_ee_rows(data_path, observations)

    raw_camera_timestamps = payload.get("camera_timestamps")
    raw_frame_source_timestamps = payload.get("camera_frame_source_timestamps")
    cameras: dict[str, CameraStream] = {}
    for camera, filename in CAMERAS.items():
        video_path = path / filename
        width, height, frames, source_fps = _video_properties(video_path)
        if raw_camera_timestamps is None:
            if frames != count:
                raise ValueError(
                    f"{video_path}: legacy episode has {frames} frames but {count} robot samples"
                )
            timestamps = robot_timestamps.copy()
        else:
            if camera not in raw_camera_timestamps:
                raise ValueError(f"{data_path}: camera_timestamps is missing {camera!r}")
            timestamps = _validate_timestamps(
                raw_camera_timestamps[camera], f"{data_path}: {camera}", frames
            )
        if raw_frame_source_timestamps is None:
            capture_timestamps = timestamps
        else:
            if camera not in raw_frame_source_timestamps:
                raise ValueError(
                    f"{data_path}: camera_frame_source_timestamps is missing {camera!r}"
                )
            capture_timestamps = _validate_timestamps(
                raw_frame_source_timestamps[camera],
                f"{data_path}: {camera} frame source",
                frames,
            )
        cameras[camera] = CameraStream(
            path=video_path,
            timestamps=timestamps,
            capture_timestamps=capture_timestamps,
            width=width,
            height=height,
            source_fps=source_fps,
        )

    overlap_start = max(
        robot_timestamps[0], *(stream.timestamps[0] for stream in cameras.values())
    )
    overlap_end = min(
        robot_timestamps[-1], *(stream.timestamps[-1] for stream in cameras.values())
    )
    if overlap_end < overlap_start:
        raise ValueError(
            f"{path}: streams have no shared time range: {overlap_start:.6f}..{overlap_end:.6f}"
        )
    target_count = int(np.floor((overlap_end - overlap_start) * target_fps + 1e-9)) + 1
    if target_count < 2:
        raise ValueError(f"{path}: shared stream duration is too short for next-step actions")
    target_source_time = overlap_start + np.arange(target_count, dtype=np.float64) / target_fps
    robot_indices = _nearest_indices(robot_timestamps, target_source_time)

    # Canonicalize after resampling as well: repeated nearest-neighbor samples
    # and skipped raw rows must not create a q/-q discontinuity in the state.
    selected_ee = measured_ee[robot_indices].copy()
    selected_ee[:, 3:7] = _canonicalize_quaternions(
        selected_ee[:, 3:7], f"{data_path}: resampled measured TCP"
    )
    state = selected_ee[:-1]
    action = _next_step_actions(selected_ee)

    camera_indices: dict[str, np.ndarray] = {}
    alignment_ms = {
        "robot": (
            np.abs(robot_timestamps[robot_indices] - target_source_time) * 1000.0
        )[:-1]
    }
    for camera, stream in cameras.items():
        indices = _nearest_indices(stream.timestamps, target_source_time)
        camera_indices[camera] = indices[:-1]
        alignment_ms[camera] = (
            np.abs(stream.capture_timestamps[indices] - target_source_time) * 1000.0
        )[:-1]

    output_count = target_count - 1
    return Episode(
        source_dir=path,
        timestamps=(np.arange(output_count, dtype=np.float64) / target_fps).astype(np.float32),
        state=state.astype(np.float32, copy=False),
        action=action,
        cameras=cameras,
        camera_indices=camera_indices,
        alignment_ms=alignment_ms,
    )


def _state_modality() -> dict[str, Any]:
    return {
        "measured_tcp_position": {
            "original_key": "observation.state",
            "start": 0,
            "end": 3,
            "absolute": True,
            "dtype": "float32",
            "unit": "meter",
            "frame": "robot_base",
        },
        "measured_tcp_quaternion_xyzw": {
            "original_key": "observation.state",
            "start": 3,
            "end": 7,
            "absolute": True,
            "dtype": "float32",
            "quaternion_order": "xyzw",
            "sign_canonicalization": "continuous_per_episode_with_nonnegative_initial_w",
        },
        "measured_gripper": {
            "original_key": "observation.state",
            "start": 7,
            "end": 8,
            "absolute": True,
            "dtype": "float32",
            "range": [0.0, 1.0],
        },
    }


def _action_modality() -> dict[str, Any]:
    return {
        "measured_tcp_translation_delta_local": {
            "original_key": "action",
            "start": 0,
            "end": 3,
            "absolute": False,
            "dtype": "float32",
            "unit": "meter",
            "frame": "measured_tcp_at_t",
        },
        "measured_tcp_rotation_delta_rotvec_local": {
            "original_key": "action",
            "start": 3,
            "end": 6,
            "absolute": False,
            "dtype": "float32",
            "unit": "radian",
            "frame": "measured_tcp_at_t",
        },
        "measured_gripper_next": {
            "original_key": "action",
            "start": 6,
            "end": 7,
            "absolute": True,
            "dtype": "float32",
            "range": [0.0, 1.0],
        },
    }


def convert(
    raw_root: Path,
    output_root: Path,
    dataset_name: str,
    task: str,
    target_fps: float,
    overwrite: bool,
) -> Path:
    raw_root = raw_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    task = task.strip()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_root}")
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("--dataset-name must be one non-empty directory name")
    if not task:
        raise ValueError("--task must not be empty")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("--target-fps must be positive")

    source_dirs = sorted({data_path.parent for data_path in raw_root.rglob("data.pkl")})
    if not source_dirs:
        raise RuntimeError(f"No episode directories containing data.pkl under {raw_root}")
    episodes = [_load_episode(path, target_fps) for path in source_dirs]
    reference = episodes[0]
    for episode in episodes[1:]:
        for camera in CAMERAS:
            current = episode.cameras[camera]
            expected = reference.cameras[camera]
            if (current.width, current.height) != (expected.width, expected.height):
                raise ValueError(
                    f"{camera} resolution changed: {(expected.width, expected.height)} vs "
                    f"{(current.width, current.height)} in {episode.source_dir}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_path = output_root / dataset_name
    if dataset_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {dataset_path} (pass --overwrite to replace it)"
        )
    staging_path = Path(tempfile.mkdtemp(prefix=f".{dataset_name}_", dir=output_root))
    try:
        total_frames = 0
        episode_lines: list[str] = []
        for episode_index, episode in enumerate(episodes):
            _write_episode_parquet(
                staging_path / f"data/chunk-000/episode_{episode_index:06d}.parquet",
                episode,
                episode_index,
                total_frames,
            )
            for camera in CAMERAS:
                destination = staging_path / (
                    f"videos/chunk-000/observation.images.{camera}/"
                    f"episode_{episode_index:06d}.mp4"
                )
                _write_resampled_video(
                    episode.cameras[camera],
                    episode.camera_indices[camera],
                    destination,
                    target_fps,
                )
            episode_lines.append(
                json.dumps(
                    {"episode_index": episode_index, "tasks": [task], "length": episode.length}
                )
            )
            total_frames += episode.length
            alignment = ", ".join(
                f"{name} median/max={np.median(error):.2f}/{np.max(error):.2f}ms"
                for name, error in episode.alignment_ms.items()
            )
            print(
                f"[{episode_index + 1}/{len(episodes)}] {episode.source_dir.name}: "
                f"{episode.length} frames at {target_fps:g} Hz; {alignment}"
            )

        info = {
            "codebase_version": "v2.1",
            "robot_type": "piper",
            "total_episodes": len(episodes),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": len(episodes) * len(CAMERAS),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": target_fps,
            "action_source": ACTION_SOURCE,
            "action_layout": ACTION_LAYOUT,
            "state_source": STATE_SOURCE,
            "state_layout": STATE_LAYOUT,
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _features(reference.cameras, target_fps, STATE_DIM),
        }
        modality = {
            "video": {
                "global": {"original_key": "observation.images.global"},
                "hand": {"original_key": "observation.images.hand"},
            },
            "state": _state_modality(),
            "action": _action_modality(),
            "annotation": {
                "human.action.task_description": {"original_key": "task_index"}
            },
        }
        embodiment = {
            "robot_name": "Piper",
            "robot_type": "piper",
            "record_frequency": target_fps,
            "body_controller_frequency": target_fps,
            "hand_controller_frequency": target_fps,
            "embodiment_tag": "new_embodiment",
            "action_source": ACTION_SOURCE,
            "action_layout": ACTION_LAYOUT,
            "state_source": STATE_SOURCE,
            "state_layout": STATE_LAYOUT,
        }
        _write_json(staging_path / "meta/info.json", info)
        _write_json(staging_path / "meta/modality.json", modality)
        _write_json(staging_path / "meta/embodiment.json", embodiment)
        (staging_path / "meta/tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": task}) + "\n", encoding="utf-8"
        )
        (staging_path / "meta/episodes.jsonl").write_text(
            "\n".join(episode_lines) + "\n", encoding="utf-8"
        )
        if dataset_path.exists():
            shutil.rmtree(dataset_path)
        staging_path.replace(dataset_path)
    except Exception:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    print(f"Converted {len(episodes)} episodes / {total_frames} frames to {dataset_path}")
    print(f"Uniform output frequency: {target_fps:g} Hz")
    print(f"Observation state: {STATE_LAYOUT}")
    print(f"Action: {ACTION_LAYOUT}")
    return dataset_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    convert(
        raw_root=args.raw_root,
        output_root=args.output_root,
        dataset_name=args.dataset_name,
        task=args.task,
        target_fps=args.target_fps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
