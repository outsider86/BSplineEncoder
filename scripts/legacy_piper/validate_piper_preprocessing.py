#!/usr/bin/env python3
"""Check unified preprocessing outputs against raw recordings, for any setup.

Example:
  python Data/RawData/validate_piper_preprocessing.py Data/StateAbsEE/*_cleanup

Reconstructs the target clock, state, action, and idle cutoff independently
from the raw pickle files listed in meta/source_episodes.jsonl. The manifest
allows moving the output dataset without depending on sorted task directories.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def vector(row: dict, fields: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(row[key], np.float32).reshape(-1) for key in fields])


def normalized_quaternions(values: np.ndarray) -> np.ndarray:
    quaternions = values.astype(np.float64).copy()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    if quaternions[0, 3] < 0:
        quaternions[0] *= -1
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return quaternions.astype(np.float32)


def nearest(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Inspect the two bracketing candidates explicitly; tuple ordering breaks
    # an equal-distance tie toward the earlier timestamp/index.
    result = []
    for target in targets:
        right = int(np.searchsorted(source, target))
        candidates = {max(0, right - 1), min(len(source) - 1, right)}
        result.append(min(candidates, key=lambda i: (abs(source[i] - target), i)))
    return np.asarray(result, dtype=np.int64)


def delta_from_matrices(ee: np.ndarray) -> np.ndarray:
    matrices = Rotation.from_quat(ee[:, 3:7].astype(np.float64)).as_matrix()
    displacement = ee[1:, :3].astype(np.float64) - ee[:-1, :3].astype(np.float64)
    inverse = np.transpose(matrices[:-1], (0, 2, 1))
    translation = np.einsum("nij,nj->ni", inverse, displacement)
    relative = inverse @ matrices[1:]
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return np.concatenate([translation, rotvec, ee[1:, 7:8]], axis=1).astype(np.float32)


def expected_from_raw(source: Path, setup: dict, run: dict):
    with (source / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    observations = payload["observations"]
    robot_time = np.asarray(payload["timestamps"], dtype=np.float64)
    require(len(observations) == len(robot_time), f"{source}: raw robot lengths differ")
    require(np.isfinite(robot_time).all() and np.all(np.diff(robot_time) >= 0), f"{source}: invalid timestamps")
    joint = np.stack([vector(row, ("arm_joint_position", "gripper_pos")) for row in observations])
    need_ee = setup["state"].startswith("ee") or "ee" in setup["action"] or run["cleanup_motion"] == "measured-ee"
    ee = None
    if need_ee:
        ee = np.stack([vector(row, ("arm_pos", "arm_quat", "gripper_pos")) for row in observations])
        if setup["quaternion"] == "continuous":
            ee[:, 3:7] = normalized_quaternions(ee[:, 3:7])
    need_command = "command" in setup["state"] or setup["action"] == "command" or run["cleanup_motion"] == "command-joint"
    commands = previous = None
    if need_command:
        commands = np.stack([vector(row, ("arm_joint_position", "gripper_pos")) for row in payload["actions"]])
        previous = np.vstack([commands[0], commands[:-1]])
    camera_times = payload.get("camera_timestamps")
    camera_times = ({camera: robot_time for camera in ("global", "hand")} if camera_times is None
                    else {camera: np.asarray(camera_times[camera], np.float64) for camera in ("global", "hand")})
    start = max(robot_time[0], *(times[0] for times in camera_times.values()))
    end = min(robot_time[-1], *(times[-1] for times in camera_times.values()))
    count = int(np.floor((end - start) * run["fps"] + 1e-9)) + 1
    target = start + np.arange(count, dtype=np.float64) / run["fps"]
    selected = nearest(robot_time, target)
    joint = joint[selected]
    if ee is not None:
        ee = ee[selected]
        if setup["quaternion"] == "continuous":
            ee[:, 3:7] = normalized_quaternions(ee[:, 3:7])
    if commands is not None:
        commands = commands[selected]
        previous = (previous[selected] if setup["last_command_clock"] == "raw"
                    else np.vstack([commands[0], commands[:-1]]))
    state_name = setup["state"]
    if state_name == "last-command":
        state = previous
    else:
        state = ee if state_name.startswith("ee") else joint
        if "last-command" in state_name:
            state = np.concatenate([state, previous], axis=1)
    offset = setup["action_offset"]
    length = count - offset
    action_name = setup["action"]
    if action_name == "delta-ee":
        action = delta_from_matrices(ee)
    elif action_name == "delta-joint":
        action = np.column_stack([joint[1:, :6] - joint[:-1, :6], joint[1:, 6]])
    elif action_name == "abs-ee":
        action = ee[offset:]
    elif action_name == "command":
        action = commands[offset:]
    else:
        action = joint[offset:]
    motion = run["cleanup_motion"]
    if motion == "auto":
        motion = {"command": "command-joint", "delta-ee": "measured-ee",
                  "delta-joint": "measured-delta-joint"}.get(action_name, "measured-joint")
    if motion == "measured-ee":
        delta = delta_from_matrices(ee)
        moving = ((np.linalg.norm(delta[:, :3], axis=1) > run["translation_threshold"])
                  | (np.linalg.norm(delta[:, 3:6], axis=1) > run["rotation_threshold"])
                  | (np.abs(ee[1:, 7] - ee[:-1, 7]) > run["gripper_threshold"]))
        motion_offset = 0
    elif motion == "measured-delta-joint":
        increments = joint[1:, :6] - joint[:-1, :6]
        moving = ((np.linalg.norm(increments, axis=1) > run["arm_threshold"])
                  | (np.abs(joint[1:, 6] - joint[:-1, 6]) > run["gripper_threshold"]))
        motion_offset = 0
    else:
        values = commands[offset:] if motion == "command-joint" else joint[:length]
        moving = ((np.linalg.norm(np.diff(values[:, :6], axis=0), axis=1) > run["arm_threshold"])
                  | (np.abs(np.diff(values[:, 6])) > run["gripper_threshold"]))
        motion_offset = 1
    cutoff = 0
    if run["cleanup"]:
        window = max(1, round(run["confirmation_seconds"] * run["fps"]))
        minimum = max(1, round(run["confirmation_fraction"] * window))
        scan = moving[:round(run["max_scan_seconds"] * run["fps"])]
        for start_index in range(len(scan) - window + 1):
            if sum(scan[start_index:start_index + window]) >= minimum:
                first = next(index for index in range(start_index, start_index + window) if scan[index])
                cutoff = max(0, first + motion_offset - round(run["preroll_seconds"] * run["fps"]))
                cutoff = min(cutoff, max(0, length - round(run["min_remaining_seconds"] * run["fps"])))
                break
    camera_indices = {camera: nearest(times, target[:length])[cutoff:] for camera, times in camera_times.items()}
    return state[:length][cutoff:], action[cutoff:], cutoff, count, camera_indices


def check_values(actual: np.ndarray, expected: np.ndarray, label: str, delta: bool = False) -> None:
    require(actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}")
    require(np.isfinite(actual).all(), f"{label}: NaN/Inf")
    equal = np.allclose(actual, expected, atol=2e-7, rtol=1e-6) if delta else np.array_equal(actual, expected)
    require(bool(equal), f"{label}: values differ, max error {np.max(np.abs(actual - expected))}")


def validate_dataset(dataset: Path, videos: bool = True) -> dict:
    info = read_json(dataset / "meta/info.json")
    config = info["preprocessing"]
    setup, run = config["setup"], config["run"]
    action_sources = {"joint": "observation", "command": "action", "abs-ee": "observation_ee",
                      "delta-joint": "observation_delta_joint", "delta-ee": "observation_delta_ee"}
    require(info["action_source"] == action_sources[setup["action"]], "Action source metadata differs")
    if setup["action"].startswith("delta-"):
        require(setup["action_offset"] == 1, "Delta targets must be t->t+1")
    require(float(info["fps"]) == run["fps"], "FPS metadata differs from configuration")
    require(info["action_timestep_offset"] == setup["action_offset"], "Action offset metadata differs")
    sources = read_jsonl(dataset / "meta/source_episodes.jsonl")
    raw_sources = sorted({path.parent.resolve() for path in Path(config["raw_root"]).rglob("data.pkl")})
    limit = config.get("limit_episodes")
    if limit is not None:
        raw_sources = raw_sources[:limit]
    require([Path(row["source_episode"]).resolve() for row in sources] == raw_sources,
            "Source manifest does not cover the expected raw episodes in order")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    with (dataset / "meta/cleanup_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    count = info["total_episodes"]
    require(len(sources) == len(episodes) == len(metrics) == count, "Episode/manifest/metric counts differ")
    require(len(list((dataset / "data").rglob("*.parquet"))) == count, "Parquet count differs")
    total = checked_videos = removed = 0
    for index, record in enumerate(sources):
        source = Path(record["source_episode"])
        state, action, cutoff, raw_count, _ = expected_from_raw(source, setup, run)
        require(record["episode_index"] == index == episodes[index]["episode_index"], "Episode order differs")
        require(record["resampled_frames"] == raw_count, f"{source}: resampled length differs")
        require(record["dropped_tail_frames"] == setup["action_offset"], f"{source}: tail drop differs")
        require(record["removed_prefix_frames"] == cutoff == int(metrics[index]["removed_frames"]), f"{source}: idle cutoff differs")
        length = len(state)
        require(length == episodes[index]["length"] == int(metrics[index]["cleaned_frames"]), "Cleaned length differs")
        require(int(metrics[index]["original_frames"]) == raw_count - setup["action_offset"], "Uncropped length differs")
        parquet = dataset / f"data/chunk-000/episode_{index:06d}.parquet"
        table = pq.read_table(parquet)
        check_values(np.asarray(table["observation.state"].to_pylist(), np.float32), state, str(parquet) + ": state")
        check_values(np.asarray(table["action"].to_pylist(), np.float32), action, str(parquet) + ": action", setup["action"] == "delta-ee")
        require(info["features"]["observation.state"]["shape"] == [state.shape[1]], "State feature dimension differs")
        require(info["features"]["action"]["shape"] == [action.shape[1]], "Action feature dimension differs")
        for column, expected in (
            ("timestamp", np.arange(length, dtype=np.float32) / run["fps"]),
            ("frame_index", np.arange(length)), ("index", np.arange(total, total + length)),
            ("episode_index", np.full(length, index)), ("task_index", np.zeros(length, np.int64)),
        ):
            require(np.array_equal(np.asarray(table[column]), expected), f"{parquet}: {column} differs")
        if videos:
            for camera in ("global", "hand"):
                video = dataset / f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4"
                capture = cv2.VideoCapture(str(video))
                try:
                    require(capture.isOpened(), f"{video}: cannot open")
                    feature = info["features"][f"observation.images.{camera}"]
                    height, width, _ = feature["shape"]
                    require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == length, f"{video}: frame count differs")
                    require(np.isclose(capture.get(cv2.CAP_PROP_FPS), run["fps"], atol=1e-3), f"{video}: FPS differs")
                    require((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (width, height), f"{video}: dimensions differ")
                finally:
                    capture.release()
                checked_videos += 1
        total += length
        removed += cutoff
    require(total == info["total_frames"], "total_frames differs")
    require(info["total_videos"] == 2 * count, "total_videos differs")
    require(len(list((dataset / "videos").rglob("*.mp4"))) == 2 * count, "Actual video count differs")
    summary = read_json(dataset / "meta/cleanup_summary.json")
    require(summary["removed_frames"] == removed, "Cleanup total differs")
    require(summary["remaining_frames"] == total, "Cleanup remaining total differs")
    modality = read_json(dataset / "meta/modality.json")
    require(modality["action"]["gripper"]["absolute"] is True,
            "Action gripper must be an absolute value")
    if setup["action"] == "delta-joint":
        joints = modality["action"]["joints"]
        require(joints["absolute"] is False and joints["unit"] == "radian",
                "Delta joint modality must describe relative angles in radians")
    elif setup["action"] == "delta-ee":
        for name in ("translation", "rotation_rotvec"):
            part = modality["action"][name]
            require(part["absolute"] is False and part["frame"] == "measured_tcp_at_t",
                    "Delta EE modality must describe a relative pose in the current TCP frame")
    result = {"passed": True, "episodes": count, "rows": total, "videos_checked": checked_videos,
              "action_offset": setup["action_offset"], "state": setup["state"], "action": setup["action"],
              "checks": ["raw_values", "action_alignment", "independent_idle_cutoff", "indices", "metadata"]}
    print(f"PASS {dataset}: {count} episodes / {total} rows / {checked_videos} videos", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        validate_dataset(dataset.resolve(), not args.skip_videos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
