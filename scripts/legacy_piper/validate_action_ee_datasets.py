#!/usr/bin/env python3
"""Validate cleaned measured-delta-EE LeRobot datasets against RawData.

The expected 30 Hz state/action rows are reconstructed independently from raw
measured TCP and gripper feedback. Commanded joint values and commanded/FK EE
poses are intentionally ignored. By default, every output MP4 is also checked.

Run from the repository root::

    conda run -n starVLA --no-capture-output \
      python Data/RawData/validate_action_ee_datasets.py
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


TASKS = ("classify_blocks", "hanging_mug", "stacking_cup")
CAMERAS = ("global", "hand")
STATE_SOURCE = "observation_ee"
ACTION_SOURCE = "observation_delta_ee"
STATE_LAYOUT = "measured_tcp_position[t] + measured_tcp_quaternion_xyzw[t] + measured_gripper[t]"
ACTION_LAYOUT = (
    "local_delta_tcp_translation[t->t+1] + local_delta_tcp_rotvec[t->t+1] "
    "+ measured_gripper[t+1]"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(
        [int(row["episode_index"]) for row in rows] == list(range(len(rows))),
        f"{path}: episode indices are not contiguous",
    )
    return rows


def _parse_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    _require(normalized in {"true", "false"}, f"{context}: invalid boolean {value!r}")
    return normalized == "true"


def _nearest(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    use_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(use_right, right, left).astype(np.int64)


def _canonicalize(quaternions: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64).copy()
    _require(values.ndim == 2 and values.shape[1] == 4, f"{context}: bad quaternion shape")
    norms = np.linalg.norm(values, axis=1)
    _require(np.isfinite(values).all() and np.all(norms > 1e-8), f"{context}: invalid quaternion")
    values /= norms[:, None]
    if values[0, 3] < 0:
        values[0] *= -1
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0:
            values[index] *= -1
    return values.astype(np.float32)


@functools.lru_cache(maxsize=None)
def _expected_episode(raw_episode: Path, fps: float) -> tuple[np.ndarray, np.ndarray, int]:
    data_path = raw_episode / "data.pkl"
    with data_path.open("rb") as stream:
        payload = pickle.load(stream)
    robot_time = np.asarray(payload["timestamps"], dtype=np.float64).reshape(-1)
    observations = payload["observations"]
    _require(len(robot_time) == len(observations) and len(robot_time) > 0, f"{data_path}: length mismatch")
    _require(np.isfinite(robot_time).all(), f"{data_path}: invalid robot timestamps")
    _require(np.all(np.diff(robot_time) >= 0), f"{data_path}: decreasing robot timestamps")

    positions: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    grippers: list[float] = []
    for index, observation in enumerate(observations):
        position = np.asarray(observation["arm_pos"], dtype=np.float32).reshape(-1)
        quaternion = np.asarray(observation["arm_quat"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(observation["gripper_pos"], dtype=np.float32).reshape(-1)
        _require(position.shape == (3,), f"{data_path}: observation {index} position shape")
        _require(quaternion.shape == (4,), f"{data_path}: observation {index} quaternion shape")
        _require(gripper.shape == (1,), f"{data_path}: observation {index} gripper shape")
        positions.append(position)
        quaternions.append(quaternion)
        grippers.append(float(gripper[0]))
    position_array = np.stack(positions)
    quaternion_array = _canonicalize(np.stack(quaternions), str(data_path))
    gripper_array = np.asarray(grippers, dtype=np.float32)[:, None]

    camera_times = payload.get("camera_timestamps")
    streams: list[np.ndarray] = []
    if camera_times is None:
        streams = [robot_time, robot_time]
    else:
        for camera in CAMERAS:
            values = np.asarray(camera_times[camera], dtype=np.float64).reshape(-1)
            _require(len(values) > 0 and np.isfinite(values).all(), f"{data_path}: invalid {camera} timestamps")
            _require(np.all(np.diff(values) >= 0), f"{data_path}: decreasing {camera} timestamps")
            streams.append(values)

    overlap_start = max(robot_time[0], *(values[0] for values in streams))
    overlap_end = min(robot_time[-1], *(values[-1] for values in streams))
    target_count = int(np.floor((overlap_end - overlap_start) * fps + 1e-9)) + 1
    _require(target_count >= 2, f"{data_path}: too short for next-step actions")
    target_time = overlap_start + np.arange(target_count, dtype=np.float64) / fps
    selected = _nearest(robot_time, target_time)
    state_all = np.concatenate(
        [position_array[selected], quaternion_array[selected], gripper_array[selected]], axis=1
    ).astype(np.float32)
    state_all[:, 3:7] = _canonicalize(state_all[:, 3:7], f"{data_path}: resampled")

    rotations = Rotation.from_quat(state_all[:, 3:7].astype(np.float64))
    local_translation = rotations[:-1].inv().apply(
        state_all[1:, :3].astype(np.float64) - state_all[:-1, :3].astype(np.float64)
    )
    local_rotvec = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
    action = np.concatenate(
        [local_translation, local_rotvec, state_all[1:, 7:8]], axis=1
    ).astype(np.float32)
    return state_all[:-1], action, target_count


def _matrix(table: Any, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), dtype=np.float32)


def _assert_exact(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    _require(actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}")
    _require(np.isfinite(actual).all(), f"{context}: NaN or Inf")
    if not np.array_equal(actual, expected):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        location = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            f"{context}: max error {error[location]:.9g} at {location}; "
            f"actual={actual[location]!r}, expected={expected[location]!r}"
        )


def _check_video(path: Path, frames: int, fps: float, size: tuple[int, int]) -> None:
    capture = cv2.VideoCapture(str(path))
    _require(capture.isOpened(), f"{path}: cannot open")
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    _require(actual_frames == frames, f"{path}: {actual_frames} frames != {frames}")
    _require(np.isclose(actual_fps, fps, atol=1e-3), f"{path}: {actual_fps:g} Hz != {fps:g}")
    _require(actual_size == size, f"{path}: size {actual_size} != {size}")


def validate(
    raw_root: Path,
    action_ee_root: Path,
    check_videos: bool,
    tasks: tuple[str, ...] = TASKS,
) -> None:
    raw_root = raw_root.resolve()
    action_ee_root = action_ee_root.resolve()
    total_frames = total_episodes = total_videos = 0
    for task in tasks:
        dataset = action_ee_root / f"{task}_30hz_cleanup"
        info = _read_json(dataset / "meta/info.json")
        fps = float(info["fps"])
        _require(np.isclose(fps, 30.0), f"{dataset}: expected 30 Hz")
        _require(info.get("state_source") == STATE_SOURCE, f"{dataset}: wrong state source")
        _require(info.get("action_source") == ACTION_SOURCE, f"{dataset}: wrong action source")
        _require(info.get("state_layout") == STATE_LAYOUT, f"{dataset}: wrong state layout")
        _require(info.get("action_layout") == ACTION_LAYOUT, f"{dataset}: wrong action layout")
        features = info["features"]
        _require(features["observation.state"]["shape"] == [8], f"{dataset}: state is not 8D")
        _require(features["action"]["shape"] == [7], f"{dataset}: action is not 7D")

        raw_episodes = sorted({path.parent for path in (raw_root / task).rglob("data.pkl")})
        metrics = _read_metrics(dataset / "meta/cleanup_metrics.csv")
        episode_meta = _read_jsonl(dataset / "meta/episodes.jsonl")
        count = int(info["total_episodes"])
        _require(len(raw_episodes) == len(metrics) == len(episode_meta) == count, f"{dataset}: episode counts differ")
        cutoffs = [int(row["removed_frames"]) for row in metrics]
        _require(len(set(cutoffs)) > 1, f"{dataset}: every episode has the same cutoff")

        global_offset = 0
        for episode_index, raw_episode in enumerate(raw_episodes):
            state, action, resampled_count = _expected_episode(raw_episode, fps)
            metric = metrics[episode_index]
            cutoff = int(metric["removed_frames"])
            _require(int(metric["original_frames"]) == resampled_count - 1, f"{dataset}: episode {episode_index} did not drop exactly the final source frame")
            expected_state = state[cutoff:]
            expected_action = action[cutoff:]
            parquet = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"
            table = pq.read_table(parquet)
            context = f"{task}/episode_{episode_index:06d}"
            _assert_exact(_matrix(table, "observation.state"), expected_state, f"{context} state")
            _assert_exact(_matrix(table, "action"), expected_action, f"{context} action")
            length = len(expected_state)
            _require(int(metric["cleaned_frames"]) == length, f"{context}: cleaned length differs")
            _require(not _parse_bool(metric["command_signal_available"], context), f"{context}: cleanup used command data")
            _require(_parse_bool(metric["measured_signal_available"], context), f"{context}: cleanup did not use measured deltas")
            _require(episode_meta[episode_index]["episode_index"] == episode_index, f"{context}: metadata index")
            _require(episode_meta[episode_index]["length"] == length, f"{context}: metadata length")
            np.testing.assert_allclose(
                np.asarray(table["timestamp"]),
                np.arange(length, dtype=np.float32) / np.float32(fps),
                rtol=0,
                atol=1e-6,
                err_msg=f"{context}: timestamp grid",
            )
            _require(np.array_equal(np.asarray(table["frame_index"]), np.arange(length)), f"{context}: frame_index")
            _require(np.array_equal(np.asarray(table["episode_index"]), np.full(length, episode_index)), f"{context}: episode_index")
            _require(np.array_equal(np.asarray(table["index"]), np.arange(global_offset, global_offset + length)), f"{context}: global index")
            _require(np.array_equal(np.asarray(table["task_index"]), np.zeros(length)), f"{context}: task index")

            if check_videos:
                for camera in CAMERAS:
                    feature = features[f"observation.images.{camera}"]
                    height, width, channels = feature["shape"]
                    _require(channels == 3, f"{dataset}: {camera} is not RGB")
                    video = dataset / (
                        f"videos/chunk-000/observation.images.{camera}/"
                        f"episode_{episode_index:06d}.mp4"
                    )
                    _check_video(video, length, fps, (width, height))
                    total_videos += 1
            global_offset += length
            total_frames += length
            total_episodes += 1

        _require(global_offset == int(info["total_frames"]), f"{dataset}: total frame count")
        _require(int(info["total_videos"]) == count * len(CAMERAS), f"{dataset}: total video count")
        print(f"PASS {task}: {count} episodes, {global_offset} frames, state=8D, action=7D")

    suffix = f", {total_videos} videos" if check_videos else " (videos skipped)"
    print(
        f"PASS all ActionEE datasets: {len(tasks)} datasets, {total_episodes} episodes, "
        f"{total_frames} rows{suffix}"
    )


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=repository_root / "Data/RawData")
    parser.add_argument(
        "--action-ee-root",
        type=Path,
        default=repository_root / "Data/ActionEE/State_EE",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--skip-videos", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate(
        args.raw_root,
        args.action_ee_root,
        not args.skip_videos,
        tuple(args.tasks),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
