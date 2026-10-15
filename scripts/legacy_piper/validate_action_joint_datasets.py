#!/usr/bin/env python3
"""Validate cleaned ActionJoint LeRobot datasets directly against RawData.

This is intentionally independent of ``convert_piper_to_lerobot.py``: it
reconstructs the 30 Hz nearest-timestamp selection from each raw ``data.pkl``
and then applies the cutoff recorded by the cleanup script.  That makes it a
useful handoff check for the exact state/action semantics, rather than a test
that merely repeats the converter's implementation.

Run from the repository root with the environment used for conversion::

    conda run -n starVLA --no-capture-output \
      python Data/RawData/validate_action_joint_datasets.py

Use ``--skip-videos`` for a faster state/action-only validation.
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
class Setting:
    directory: str
    state_source: str
    state_dim: int
    state_layout: str


SETTINGS = (
    Setting(
        "State_LastCommand_Joint",
        "state_last_command",
        14,
        "measured_state[t] + command[t-1] (command[0] at boundary)",
    ),
    Setting("State_Joint", "state", 7, "measured_state[t]"),
    Setting(
        "LastCommand_Joint",
        "last_command",
        7,
        "command[t-1] (command[0] at boundary)",
    ),
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


def _read_cleanup_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indices = [int(row["episode_index"]) for row in rows]
    _require(indices == list(range(len(rows))), f"{path}: non-contiguous episode indices")
    return rows


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Independent nearest-neighbor timestamp lookup; ties select the left sample."""
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, len(source) - 1)
    left = np.maximum(right - 1, 0)
    use_right = np.abs(source[right] - target) < np.abs(target - source[left])
    return np.where(use_right, right, left).astype(np.int64)


def _raw_vector(row: dict[str, Any], label: str, source: Path) -> np.ndarray:
    try:
        vector = np.concatenate(
            [
                np.asarray(row["arm_joint_position"], dtype=np.float32).reshape(-1),
                np.asarray(row["gripper_pos"], dtype=np.float32).reshape(-1),
            ]
        )
    except KeyError as error:
        raise AssertionError(f"{source}: {label} is missing {error.args[0]!r}") from error
    _require(vector.shape == (7,), f"{source}: {label} has shape {vector.shape}, expected (7,)")
    return vector


@functools.lru_cache(maxsize=None)
def _expected_from_raw(raw_episode: Path, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return independently reconstructed measured[t] and raw command[t-1]."""
    data_path = raw_episode / "data.pkl"
    with data_path.open("rb") as stream:
        payload = pickle.load(stream)

    robot_time = np.asarray(payload["timestamps"], dtype=np.float64).reshape(-1)
    observations = payload["observations"]
    commands = payload["actions"]
    _require(len(robot_time) > 0, f"{data_path}: empty robot stream")
    _require(
        len(observations) == len(robot_time) == len(commands),
        f"{data_path}: robot timestamp/observation/action lengths differ",
    )
    _require(np.isfinite(robot_time).all(), f"{data_path}: non-finite robot timestamp")
    _require(np.all(np.diff(robot_time) >= 0), f"{data_path}: robot timestamps decrease")

    measured = np.stack(
        [_raw_vector(row, f"observation[{index}]", data_path) for index, row in enumerate(observations)]
    )
    command = np.stack(
        [_raw_vector(row, f"action[{index}]", data_path) for index, row in enumerate(commands)]
    )
    previous_command = np.empty_like(command)
    previous_command[0] = command[0]
    previous_command[1:] = command[:-1]

    camera_times = payload.get("camera_timestamps")
    if camera_times is None:
        # Legacy recordings stored one camera frame for every robot sample.
        stream_times = [robot_time, robot_time]
    else:
        stream_times = []
        for camera in CAMERAS:
            _require(camera in camera_times, f"{data_path}: missing {camera} camera timestamps")
            values = np.asarray(camera_times[camera], dtype=np.float64).reshape(-1)
            _require(len(values) > 0, f"{data_path}: empty {camera} camera timestamps")
            _require(np.isfinite(values).all(), f"{data_path}: non-finite {camera} timestamp")
            _require(np.all(np.diff(values) >= 0), f"{data_path}: {camera} timestamps decrease")
            stream_times.append(values)

    overlap_start = max(robot_time[0], *(values[0] for values in stream_times))
    overlap_end = min(robot_time[-1], *(values[-1] for values in stream_times))
    _require(overlap_end >= overlap_start, f"{data_path}: streams do not overlap")
    frame_count = int(np.floor((overlap_end - overlap_start) * fps + 1e-9)) + 1
    target_time = overlap_start + np.arange(frame_count, dtype=np.float64) / fps
    selected = _nearest_indices(robot_time, target_time)
    return measured[selected], previous_command[selected]


def _matrix(table: Any, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _assert_equal_float(actual: np.ndarray, expected: np.ndarray, context: str) -> None:
    _require(actual.shape == expected.shape, f"{context}: shape {actual.shape} != {expected.shape}")
    _require(np.isfinite(actual).all(), f"{context}: output contains NaN or Inf")
    if not np.array_equal(actual, expected):
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        location = np.unravel_index(int(np.argmax(error)), error.shape)
        raise AssertionError(
            f"{context}: values differ; max_abs_error={error[location]:.9g} at {location}, "
            f"actual={actual[location]!r}, expected={expected[location]!r}"
        )


def _parse_bool(value: str, context: str) -> bool:
    normalized = value.strip().lower()
    _require(normalized in {"true", "false"}, f"{context}: expected boolean, got {value!r}")
    return normalized == "true"


def _check_video(path: Path, frames: int, fps: float, size: tuple[int, int]) -> None:
    capture = cv2.VideoCapture(str(path))
    _require(capture.isOpened(), f"{path}: cannot open video")
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_size = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    _require(actual_frames == frames, f"{path}: {actual_frames} frames, expected {frames}")
    _require(np.isclose(actual_fps, fps, atol=1e-3), f"{path}: {actual_fps:g} Hz, expected {fps:g}")
    _require(actual_size == size, f"{path}: size {actual_size}, expected {size}")


def validate(raw_root: Path, action_joint_root: Path, check_videos: bool) -> None:
    raw_root = raw_root.resolve()
    action_joint_root = action_joint_root.resolve()

    metrics_by_setting: dict[tuple[str, str], list[dict[str, str]]] = {}
    for setting in SETTINGS:
        for task in TASKS:
            dataset = action_joint_root / setting.directory / f"{task}_30hz_cleanup"
            metrics_by_setting[(setting.directory, task)] = _read_cleanup_metrics(
                dataset / "meta/cleanup_metrics.csv"
            )

    # Cleanup must select the same physical clip regardless of state layout.
    for task in TASKS:
        reference = metrics_by_setting[(SETTINGS[0].directory, task)]
        reference_cutoffs = [int(row["removed_frames"]) for row in reference]
        _require(len(set(reference_cutoffs)) > 1, f"{task}: all episodes use one identical cutoff")
        for setting in SETTINGS[1:]:
            candidate = metrics_by_setting[(setting.directory, task)]
            candidate_cutoffs = [int(row["removed_frames"]) for row in candidate]
            _require(
                candidate_cutoffs == reference_cutoffs,
                f"{task}: cleanup cutoffs differ for {setting.directory}",
            )

    checked_frames = 0
    checked_episodes = 0
    checked_videos = 0
    for setting in SETTINGS:
        for task in TASKS:
            dataset = action_joint_root / setting.directory / f"{task}_30hz_cleanup"
            info = _read_json(dataset / "meta/info.json")
            fps = float(info["fps"])
            _require(np.isclose(fps, 30.0), f"{dataset}: fps={fps:g}, expected 30")
            _require(info.get("action_source") == "observation", f"{dataset}: action is not measured feedback")
            _require(info.get("state_source") == setting.state_source, f"{dataset}: wrong state_source")
            _require(info.get("state_layout") == setting.state_layout, f"{dataset}: wrong state_layout")
            features = info["features"]
            _require(features["observation.state"]["shape"] == [setting.state_dim], f"{dataset}: wrong state dimension")
            _require(features["action"]["shape"] == [7], f"{dataset}: wrong action dimension")

            raw_episodes = sorted({path.parent for path in (raw_root / task).rglob("data.pkl")})
            metrics = metrics_by_setting[(setting.directory, task)]
            episode_meta = _read_jsonl(dataset / "meta/episodes.jsonl")
            expected_episodes = int(info["total_episodes"])
            _require(len(raw_episodes) == expected_episodes, f"{dataset}: RawData episode count differs")
            _require(len(metrics) == expected_episodes, f"{dataset}: cleanup metric count differs")
            _require(len(episode_meta) == expected_episodes, f"{dataset}: episode metadata count differs")

            global_offset = 0
            for episode_index, raw_episode in enumerate(raw_episodes):
                measured, previous_command = _expected_from_raw(raw_episode, fps)
                metric = metrics[episode_index]
                cutoff = int(metric["removed_frames"])
                _require(int(metric["original_frames"]) == len(measured), f"{dataset}: episode {episode_index} original length differs")
                expected_action = measured[cutoff:]
                if setting.state_source == "state":
                    expected_state = expected_action
                elif setting.state_source == "last_command":
                    expected_state = previous_command[cutoff:]
                else:
                    expected_state = np.concatenate(
                        [expected_action, previous_command[cutoff:]], axis=1
                    )

                parquet_path = dataset / f"data/chunk-000/episode_{episode_index:06d}.parquet"
                table = pq.read_table(parquet_path)
                state = _matrix(table, "observation.state")
                action = _matrix(table, "action")
                context = f"{setting.directory}/{task}/episode_{episode_index:06d}"
                _assert_equal_float(action, expected_action, f"{context} action")
                _assert_equal_float(state, expected_state, f"{context} state")

                count = len(expected_action)
                _require(int(metric["cleaned_frames"]) == count, f"{context}: cleanup length differs")
                _require(episode_meta[episode_index]["episode_index"] == episode_index, f"{context}: bad episode metadata index")
                _require(episode_meta[episode_index]["length"] == count, f"{context}: bad episode metadata length")
                np.testing.assert_allclose(
                    np.asarray(table["timestamp"]),
                    np.arange(count, dtype=np.float32) / np.float32(fps),
                    rtol=0,
                    atol=1e-6,
                    err_msg=f"{context}: timestamps are not a reset 30 Hz clock",
                )
                _require(np.array_equal(np.asarray(table["frame_index"]), np.arange(count)), f"{context}: frame_index differs")
                _require(np.array_equal(np.asarray(table["episode_index"]), np.full(count, episode_index)), f"{context}: episode_index differs")
                _require(np.array_equal(np.asarray(table["index"]), np.arange(global_offset, global_offset + count)), f"{context}: global index differs")
                _require(np.array_equal(np.asarray(table["task_index"]), np.zeros(count)), f"{context}: task_index differs")

                expected_command_available = setting.state_source != "state"
                _require(
                    _parse_bool(metric["command_signal_available"], context) == expected_command_available,
                    f"{context}: cleanup command availability is wrong",
                )
                _require(
                    _parse_bool(metric["measured_signal_available"], context),
                    f"{context}: cleanup did not see measured action feedback",
                )

                if check_videos:
                    for camera in CAMERAS:
                        video_feature = features[f"observation.images.{camera}"]
                        height, width, channels = video_feature["shape"]
                        _require(channels == 3, f"{dataset}: {camera} is not RGB")
                        video_path = dataset / (
                            f"videos/chunk-000/observation.images.{camera}/"
                            f"episode_{episode_index:06d}.mp4"
                        )
                        _check_video(video_path, count, fps, (width, height))
                        checked_videos += 1

                global_offset += count
                checked_episodes += 1
                checked_frames += count

            _require(global_offset == int(info["total_frames"]), f"{dataset}: total_frames differs")
            _require(int(info["total_videos"]) == expected_episodes * len(CAMERAS), f"{dataset}: total_videos differs")
            print(
                f"PASS {setting.directory}/{task}: {expected_episodes} episodes, "
                f"{global_offset} frames, state={setting.state_dim}D"
            )

    suffix = f", {checked_videos} videos" if check_videos else " (videos skipped)"
    print(
        f"PASS all datasets: {len(SETTINGS) * len(TASKS)} datasets, "
        f"{checked_episodes} episode copies, {checked_frames} frame rows{suffix}"
    )


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=repository_root / "Data/RawData",
        help="RawData directory containing the three task folders.",
    )
    parser.add_argument(
        "--action-joint-root",
        type=Path,
        default=repository_root / "Data/ActionJoint",
        help="Directory containing the three cleaned state-setting folders.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip MP4 frame-count/fps/resolution checks.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate(args.raw_root, args.action_joint_root, not args.skip_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
