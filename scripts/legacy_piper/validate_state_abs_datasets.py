#!/usr/bin/env python3
"""Validate StateAbsEE and StateAbsJoint datasets directly against RawData.

The validator independently rebuilds the 30 Hz nearest-timestamp selection,
independently recomputes every idle-prefix cutoff, and checks every retained
state/action value, index, timestamp, metadata record, and video.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq


TASKS = ("classify_blocks", "hanging_mug", "stacking_cup")
CAMERAS = ("global", "hand")


@dataclass(frozen=True)
class Setup:
    root_name: str
    action_source: str
    action_dim: int
    action_layout: str


SETUPS = (
    Setup(
        "StateAbsEE",
        "observation_ee",
        8,
        "measured absolute TCP position xyz + quaternion xyzw + measured gripper at t+1",
    ),
    Setup(
        "StateAbsJoint",
        "observation",
        7,
        "measured joints + measured gripper at t+1",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(
        [int(row["episode_index"]) for row in rows] == list(range(len(rows))),
        f"{path}: episode indices are not contiguous",
    )
    return rows


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    choose_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(choose_right, right, left).astype(np.int64)


def _vector(row: dict[str, Any], keys: tuple[str, ...], path: Path) -> np.ndarray:
    try:
        result = np.concatenate(
            [np.asarray(row[key], dtype=np.float32).reshape(-1) for key in keys]
        )
    except KeyError as error:
        raise AssertionError(f"{path}: missing raw field {error.args[0]!r}") from error
    _require(np.isfinite(result).all(), f"{path}: raw vector contains NaN/Inf")
    return result


@functools.lru_cache(maxsize=None)
def _expected_episode(raw_episode: Path, fps: float) -> tuple[np.ndarray, np.ndarray]:
    data_path = raw_episode / "data.pkl"
    with data_path.open("rb") as stream:
        payload = pickle.load(stream)
    robot_time = np.asarray(payload["timestamps"], dtype=np.float64).reshape(-1)
    observations = payload["observations"]
    _require(len(robot_time) == len(observations) > 0, f"{data_path}: invalid robot stream")
    _require(np.isfinite(robot_time).all(), f"{data_path}: non-finite robot timestamps")
    _require(np.all(np.diff(robot_time) >= 0), f"{data_path}: decreasing robot timestamps")

    measured_joint = np.stack(
        [_vector(row, ("arm_joint_position", "gripper_pos"), data_path) for row in observations]
    )
    measured_ee = np.stack(
        [_vector(row, ("arm_pos", "arm_quat", "gripper_pos"), data_path) for row in observations]
    )
    _require(measured_joint.shape[1] == 7, f"{data_path}: joint state is not 7D")
    _require(measured_ee.shape[1] == 8, f"{data_path}: EE state is not 8D")

    raw_camera_time = payload.get("camera_timestamps")
    if raw_camera_time is None:
        camera_times = [robot_time, robot_time]
    else:
        camera_times = []
        for camera in CAMERAS:
            values = np.asarray(raw_camera_time[camera], dtype=np.float64).reshape(-1)
            _require(len(values) > 0, f"{data_path}: empty {camera} timestamps")
            _require(np.isfinite(values).all(), f"{data_path}: invalid {camera} timestamps")
            _require(np.all(np.diff(values) >= 0), f"{data_path}: decreasing {camera} timestamps")
            camera_times.append(values)
    overlap_start = max(robot_time[0], *(values[0] for values in camera_times))
    overlap_end = min(robot_time[-1], *(values[-1] for values in camera_times))
    _require(overlap_end >= overlap_start, f"{data_path}: streams do not overlap")
    count = int(np.floor((overlap_end - overlap_start) * fps + 1e-9)) + 1
    target = overlap_start + np.arange(count, dtype=np.float64) / fps
    selected = _nearest_indices(robot_time, target)
    return measured_joint[selected], measured_ee[selected]


def _expected_cutoff(state: np.ndarray, config: dict[str, Any]) -> int:
    arm_delta = np.linalg.norm(np.diff(state[:, :6], axis=0), axis=1)
    gripper_delta = np.abs(np.diff(state[:, 6], axis=0))
    moving = (
        (arm_delta > float(config["arm_delta_threshold"]))
        | (gripper_delta > float(config["gripper_delta_threshold"]))
    )
    scan = moving[: int(config["max_scan_frames"])]
    window = int(config["confirmation_window_frames"])
    if len(scan) < window:
        return 0
    counts = np.convolve(scan.astype(np.int32), np.ones(window, np.int32), mode="valid")
    candidates = np.flatnonzero(counts >= int(config["confirmation_min_moving_frames"]))
    if not len(candidates):
        return 0
    window_start = int(candidates[0])
    first_moving = int(np.flatnonzero(scan[window_start : window_start + window])[0])
    onset_frame = window_start + first_moving + 1
    cutoff = max(0, onset_frame - int(config["preroll_frames"]))
    return min(cutoff, max(0, len(state) - int(config["min_remaining_frames"])))


def _matrix(table: Any, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _exact(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    _require(actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}")
    _require(np.isfinite(actual).all(), f"{context}: contains NaN/Inf")
    if not np.array_equal(actual, expected):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        where = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            f"{context}: max error {error[where]:.9g} at {where}; "
            f"actual={actual[where]!r}, expected={expected[where]!r}"
        )


def _check_video(path: Path, frames: int, fps: float, size: tuple[int, int]) -> None:
    capture = cv2.VideoCapture(str(path))
    _require(capture.isOpened(), f"{path}: cannot open")
    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    _require(actual[0] == frames, f"{path}: {actual[0]} frames, expected {frames}")
    _require(np.isclose(actual[1], fps, atol=1e-3), f"{path}: {actual[1]} Hz, expected {fps}")
    _require(actual[2:] == size, f"{path}: size {actual[2:]}, expected {size}")


def validate(data_root: Path, raw_root: Path, tasks: tuple[str, ...], videos: bool) -> None:
    data_root = data_root.resolve()
    raw_root = raw_root.resolve()
    checked_rows = checked_episodes = checked_videos = 0

    cutoffs: dict[tuple[str, str], list[int]] = {}
    for setup in SETUPS:
        for task in tasks:
            dataset = data_root / setup.root_name / f"{task}_30hz_cleanup"
            metrics = _read_metrics(dataset / "meta/cleanup_metrics.csv")
            cutoffs[(setup.root_name, task)] = [int(row["removed_frames"]) for row in metrics]
    for task in tasks:
        ee = cutoffs[("StateAbsEE", task)]
        joint = cutoffs[("StateAbsJoint", task)]
        _require(ee == joint, f"{task}: the two setups retained different clips")
        _require(sum(ee) > 0, f"{task}: no leading idle frames were removed")

    for setup in SETUPS:
        for task in tasks:
            dataset = data_root / setup.root_name / f"{task}_30hz_cleanup"
            info = _read_json(dataset / "meta/info.json")
            episodes_meta = _read_jsonl(dataset / "meta/episodes.jsonl")
            metrics = _read_metrics(dataset / "meta/cleanup_metrics.csv")
            summary = _read_json(dataset / "meta/cleanup_summary.json")
            fps = float(info["fps"])
            _require(np.isclose(fps, 30.0), f"{dataset}: fps is not 30")
            _require(info.get("state_source") == "state", f"{dataset}: wrong state source")
            _require(info.get("state_layout") == "measured_state[t]", f"{dataset}: wrong state layout")
            _require(info.get("action_source") == setup.action_source, f"{dataset}: wrong action source")
            _require(info.get("action_timestep_offset") == 1, f"{dataset}: action is not next-step")
            _require(info.get("action_layout") == setup.action_layout, f"{dataset}: wrong action layout")
            _require(info["features"]["observation.state"]["shape"] == [7], f"{dataset}: state is not 7D")
            _require(info["features"]["action"]["shape"] == [setup.action_dim], f"{dataset}: wrong action dimension")

            raw_episodes = sorted({path.parent for path in (raw_root / task).rglob("data.pkl")})
            count = int(info["total_episodes"])
            _require(len(raw_episodes) == count, f"{dataset}: raw episode count mismatch")
            _require(len(episodes_meta) == count == len(metrics), f"{dataset}: metadata count mismatch")
            config = summary["config"]
            global_offset = 0
            dataset_rows = 0
            for episode_index, raw_episode in enumerate(raw_episodes):
                measured_joint, measured_ee = _expected_episode(raw_episode, fps)
                # The converter drops the last current frame because no t+1
                # target exists. Cleanup therefore operates on this prefix.
                converted_state = measured_joint[:-1]
                expected_cutoff = _expected_cutoff(converted_state, config)
                metric = metrics[episode_index]
                cutoff = int(metric["removed_frames"])
                _require(cutoff == expected_cutoff, f"{dataset}: episode {episode_index} cutoff {cutoff} != independently expected {expected_cutoff}")
                _require(int(metric["original_frames"]) == len(converted_state), f"{dataset}: episode {episode_index} original length mismatch")
                expected_state = converted_state[cutoff:]
                next_target = measured_ee[1:] if setup.action_dim == 8 else measured_joint[1:]
                expected_action = next_target[cutoff:]

                parquet = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"
                table = pq.read_table(parquet)
                _exact(_matrix(table, "observation.state"), expected_state, f"{parquet}: state")
                _exact(_matrix(table, "action"), expected_action, f"{parquet}: action")
                length = len(expected_state)
                _exact(np.asarray(table["timestamp"], np.float32), np.arange(length, dtype=np.float32) / fps, f"{parquet}: timestamp")
                _require(np.array_equal(np.asarray(table["frame_index"]), np.arange(length)), f"{parquet}: frame_index mismatch")
                _require(np.array_equal(np.asarray(table["episode_index"]), np.full(length, episode_index)), f"{parquet}: episode_index mismatch")
                _require(np.array_equal(np.asarray(table["index"]), np.arange(global_offset, global_offset + length)), f"{parquet}: global index mismatch")
                _require(episodes_meta[episode_index]["length"] == length, f"{dataset}: episode length metadata mismatch")

                if videos:
                    for camera in CAMERAS:
                        feature = info["features"][f"observation.images.{camera}"]
                        height, width, _ = feature["shape"]
                        path = dataset / f"videos/chunk-000/observation.images.{camera}/episode_{episode_index:06d}.mp4"
                        _check_video(path, length, fps, (width, height))
                        checked_videos += 1
                global_offset += length
                dataset_rows += length
                checked_episodes += 1
            _require(dataset_rows == int(info["total_frames"]), f"{dataset}: total_frames mismatch")
            _require(dataset_rows == int(summary["remaining_frames"]), f"{dataset}: cleanup summary mismatch")
            checked_rows += dataset_rows
            print(f"PASS {setup.root_name}/{task}: {count} episodes, {dataset_rows} rows")

    print(
        f"PASS all: {checked_episodes} dataset-episodes, {checked_rows} rows"
        + (f", {checked_videos} videos" if videos else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--raw-root", type=Path, default=Path("Data/RawData"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    validate(args.data_root, args.raw_root, tuple(args.tasks), not args.skip_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
