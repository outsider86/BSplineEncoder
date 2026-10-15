"""Independent validation of an exported dataset against raw recordings.

Ported from ``scripts/legacy_piper/validate_piper_preprocessing.py`` and generalized to
camera subsets, extra observations, unit conversion, and both delta-EE
frames.  The expected rows are rebuilt from ``data.pkl`` with separate code
(explicit two-candidate nearest search, rotation matrices), so the check does
not merely repeat the exporter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _vector(row: dict, fields: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([np.asarray(row[key], np.float32).reshape(-1) for key in fields])


def _normalized(values: np.ndarray) -> np.ndarray:
    quaternions = values.astype(np.float64).copy()
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    if quaternions[0, 3] < 0:
        quaternions[0] *= -1
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return quaternions.astype(np.float32)


def _nearest(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    result = []
    for target in targets:
        right = int(np.searchsorted(source, target))
        candidates = {max(0, right - 1), min(len(source) - 1, right)}
        result.append(min(candidates, key=lambda i: (abs(source[i] - target), i)))
    return np.asarray(result, dtype=np.int64)


def _delta(ee: np.ndarray, frame: str) -> np.ndarray:
    matrices = Rotation.from_quat(ee[:, 3:7].astype(np.float64)).as_matrix()
    displacement = ee[1:, :3].astype(np.float64) - ee[:-1, :3].astype(np.float64)
    if frame == "tool":
        inverse = np.transpose(matrices[:-1], (0, 2, 1))
        translation = np.einsum("nij,nj->ni", inverse, displacement)
        relative = inverse @ matrices[1:]
    else:
        translation = displacement
        relative = matrices[1:] @ np.transpose(matrices[:-1], (0, 2, 1))
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return np.concatenate([translation, rotvec, ee[1:, 7:8]], axis=1).astype(np.float32)


def _units(values: np.ndarray, kinds: list[tuple[int, int, str]], setup: dict) -> np.ndarray:
    if setup.get("angle_unit", "radian") == "radian" and setup.get("position_unit", "meter") == "meter":
        return values
    values = values.astype(np.float64).copy()
    for start, end, kind in kinds:
        if kind == "angle" and setup["angle_unit"] == "degree":
            values[:, start:end] *= 180.0 / np.pi
        if kind == "position" and setup["position_unit"] == "millimeter":
            values[:, start:end] *= 1000.0
    return values.astype(np.float32)


def _kinds(setup: dict, part: str) -> list[tuple[int, int, str]]:
    if part == "state":
        name = setup["state"]
        kinds, prefix = [], 0
        if name.startswith("joint"):
            kinds, prefix = [(0, 6, "angle")], 7
        elif name.startswith("ee"):
            kinds, prefix = [(0, 3, "position")], 8
        if "command" in name:
            kinds.append((prefix, prefix + 6, "angle"))
        return kinds
    action = setup["action"]
    if action in ("joint", "command", "delta-joint"):
        return [(0, 6, "angle")]
    if action == "abs-ee":
        return [(0, 3, "position")]
    return [(0, 3, "position"), (3, 6, "angle")]


def expected_from_raw(source: Path, setup: dict, run: dict, cameras: list[dict] | None = None,
                      extras: list[dict] | None = None):
    """Rebuild ``(state, action, cutoff, resampled_count, camera_indices, extras)``."""

    cameras = ([{"name": n, "timestamps_key": n} for n in ("global", "hand")]
               if cameras is None else cameras)
    extras = extras or []
    with (Path(source) / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    observations = payload["observations"]
    robot_time = np.asarray(payload["timestamps"], dtype=np.float64)
    require(len(observations) == len(robot_time), f"{source}: raw robot lengths differ")
    require(np.isfinite(robot_time).all() and np.all(np.diff(robot_time) >= 0),
            f"{source}: invalid timestamps")
    joint = np.stack([_vector(row, ("arm_joint_position", "gripper_pos")) for row in observations])
    need_ee = (setup["state"].startswith("ee") or "ee" in setup["action"]
               or run["cleanup_motion"] == "measured-ee")
    ee = None
    if need_ee:
        ee = np.stack([_vector(row, ("arm_pos", "arm_quat", "gripper_pos")) for row in observations])
        if setup["quaternion"] == "continuous":
            ee[:, 3:7] = _normalized(ee[:, 3:7])
    need_command = ("command" in setup["state"] or setup["action"] == "command"
                    or run["cleanup_motion"] == "command-joint")
    commands = previous = None
    if need_command:
        commands = np.stack([_vector(row, ("arm_joint_position", "gripper_pos"))
                             for row in payload["actions"]])
        previous = np.vstack([commands[0], commands[:-1]])
    raw_camera_times = payload.get("camera_timestamps")
    camera_times = {
        camera["name"]: (robot_time if raw_camera_times is None
                         else np.asarray(raw_camera_times[camera["timestamps_key"]], np.float64))
        for camera in cameras
    }
    start = max([robot_time[0], *(times[0] for times in camera_times.values())])
    end = min([robot_time[-1], *(times[-1] for times in camera_times.values())])
    count = int(np.floor((end - start) * run["fps"] + 1e-9)) + 1
    target = start + np.arange(count, dtype=np.float64) / run["fps"]
    selected = _nearest(robot_time, target)
    joint = joint[selected]
    if ee is not None:
        ee = ee[selected]
        if setup["quaternion"] == "continuous":
            ee[:, 3:7] = _normalized(ee[:, 3:7])
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
        action = _delta(ee, setup.get("delta_ee_frame", "tool"))
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
        delta = _delta(ee, "tool")
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
                first = next(i for i in range(start_index, start_index + window) if scan[i])
                cutoff = max(0, first + motion_offset - round(run["preroll_seconds"] * run["fps"]))
                cutoff = min(cutoff, max(0, length - round(run["min_remaining_seconds"] * run["fps"])))
                break
    camera_indices = {name: _nearest(times, target[:length])[cutoff:]
                      for name, times in camera_times.items()}
    extra_values = {
        extra["key"]: np.stack([np.asarray(row[extra["field"]], np.float32).reshape(-1)
                                for row in observations])[selected][:length][cutoff:]
        for extra in extras
    }
    state = _units(state[:length][cutoff:], _kinds(setup, "state"), setup)
    action = _units(action[cutoff:], _kinds(setup, "action"), setup)
    return state, action, cutoff, count, camera_indices, extra_values


def _check(actual: np.ndarray, expected: np.ndarray, label: str, approx: bool) -> None:
    require(actual.shape == expected.shape, f"{label}: {actual.shape} != {expected.shape}")
    require(np.isfinite(actual).all(), f"{label}: NaN/Inf")
    equal = (np.allclose(actual, expected, atol=2e-7, rtol=1e-6) if approx
             else np.array_equal(actual, expected))
    require(bool(equal), f"{label}: values differ, max error {np.max(np.abs(actual - expected))}")


def _matrix(table, name: str) -> np.ndarray:
    return np.asarray(table[name].to_pylist(), np.float32)


def validate_dataset(dataset: Path, videos: bool = True) -> dict:
    """Raise ``AssertionError`` on the first mismatch; return a summary dict."""

    import pyarrow.parquet as pq

    dataset = Path(dataset)
    info = read_json(dataset / "meta/info.json")
    require(info["codebase_version"] == "v2.1", "codebase_version must be v2.1")
    config = info["preprocessing"]
    setup, run = config["setup"], config["run"]
    cameras = config.get("cameras", [{"name": n, "key": f"observation.images.{n}",
                                      "timestamps_key": n} for n in ("global", "hand")])
    extras = config.get("extra_observations", [])
    action_sources = {"joint": "observation", "command": "action", "abs-ee": "observation_ee",
                      "delta-joint": "observation_delta_joint",
                      "delta-ee": "observation_delta_ee"}
    require(info["action_source"] == action_sources[setup["action"]], "Action source metadata differs")
    if setup["action"].startswith("delta-"):
        require(setup["action_offset"] == 1, "Delta targets must be t->t+1")
    require(float(info["fps"]) == run["fps"], "FPS metadata differs from configuration")
    require(info["action_timestep_offset"] == setup["action_offset"], "Action offset metadata differs")
    sources = read_jsonl(dataset / "meta/source_episodes.jsonl")
    episodes = read_jsonl(dataset / "meta/episodes.jsonl")
    episode_stats = read_jsonl(dataset / "meta/episodes_stats.jsonl")
    tasks = read_jsonl(dataset / "meta/tasks.jsonl")
    with (dataset / "meta/cleanup_metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    count = info["total_episodes"]
    require(len(sources) == len(episodes) == len(metrics) == len(episode_stats) == count,
            "Episode/manifest/metric/stat counts differ")
    require(len(tasks) == info["total_tasks"] == 1, "Task metadata differs")
    require(len(list((dataset / "data").rglob("*.parquet"))) == count, "Parquet count differs")
    video_keys = sorted(key for key, value in info["features"].items() if value["dtype"] == "video")
    require(video_keys == sorted(camera["key"] for camera in cameras),
            "Video features differ from the selected cameras")
    require((info["video_path"] is None) == (not video_keys), "video_path must be null without cameras")
    total = checked_videos = removed = 0
    for index, record in enumerate(sources):
        source = Path(record["source_episode"])
        state, action, cutoff, raw_count, camera_indices, extra_values = expected_from_raw(
            source, setup, run, cameras, extras)
        chunk = index // info["chunks_size"]
        require(record["episode_index"] == index == episodes[index]["episode_index"]
                == episode_stats[index]["episode_index"], "Episode order differs")
        require(record["resampled_frames"] == raw_count, f"{source}: resampled length differs")
        require(record["dropped_tail_frames"] == setup["action_offset"], f"{source}: tail drop differs")
        require(record["removed_prefix_frames"] == cutoff == int(metrics[index]["removed_frames"]),
                f"{source}: idle cutoff differs")
        length = len(state)
        require(length == episodes[index]["length"] == int(metrics[index]["cleaned_frames"]),
                "Cleaned length differs")
        require(int(metrics[index]["original_frames"]) == raw_count - setup["action_offset"],
                "Uncropped length differs")
        parquet = dataset / info["data_path"].format(episode_chunk=chunk, episode_index=index)
        table = pq.read_table(parquet)
        require(set(table.column_names) == {key for key, value in info["features"].items()
                                            if value["dtype"] != "video"},
                f"{parquet}: columns differ from non-video features")
        converted = (setup.get("angle_unit", "radian") != "radian"
                     or setup.get("position_unit", "meter") != "meter")
        _check(_matrix(table, "observation.state"), state, f"{parquet}: state", converted)
        _check(_matrix(table, "action"), action, f"{parquet}: action",
               converted or setup["action"] == "delta-ee")
        for key, values in extra_values.items():
            _check(_matrix(table, key), values, f"{parquet}: {key}", False)
        require(info["features"]["observation.state"]["shape"] == [state.shape[1]],
                "State feature dimension differs")
        require(info["features"]["action"]["shape"] == [action.shape[1]],
                "Action feature dimension differs")
        for column, expected in (
            ("timestamp", np.arange(length, dtype=np.float32) / run["fps"]),
            ("frame_index", np.arange(length)), ("index", np.arange(total, total + length)),
            ("episode_index", np.full(length, index)), ("task_index", np.zeros(length, np.int64)),
        ):
            require(np.array_equal(np.asarray(table[column]), expected), f"{parquet}: {column} differs")
        stats = episode_stats[index]["stats"]
        for key in table.column_names:
            values = np.asarray(table[key].to_pylist(), np.float64)
            values = values[:, None] if values.ndim == 1 else values
            require(np.allclose(stats[key]["mean"], values.mean(0), rtol=1e-6, atol=1e-6)
                    and np.allclose(stats[key]["min"], values.min(0))
                    and np.allclose(stats[key]["max"], values.max(0))
                    and stats[key]["count"] == [length], f"{parquet}: stats for {key} differ")
        alignment = pq.read_table(dataset / f"meta/source_alignment/chunk-{chunk:03d}/"
                                  f"episode_{index:06d}.parquet")
        require(np.array_equal(np.asarray(alignment["frame_index"]), np.arange(length)),
                "Source alignment frame index differs")
        for camera in cameras:
            name = camera["name"]
            require(np.array_equal(np.asarray(alignment[f"camera.{name}.source_frame_index"]),
                                   camera_indices[name]), f"{source}: {name} frame selection differs")
            if videos:
                import cv2

                video = dataset / info["video_path"].format(
                    episode_chunk=chunk, video_key=camera["key"], episode_index=index)
                capture = cv2.VideoCapture(str(video))
                try:
                    require(capture.isOpened(), f"{video}: cannot open")
                    height, width, _ = info["features"][camera["key"]]["shape"]
                    require(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == length,
                            f"{video}: frame count differs")
                    require(np.isclose(capture.get(cv2.CAP_PROP_FPS), run["fps"], atol=1e-3),
                            f"{video}: FPS differs")
                    require((int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (width, height),
                            f"{video}: dimensions differ")
                finally:
                    capture.release()
                require(stats[camera["key"]]["count"][0] > 0, f"{video}: missing image stats")
                checked_videos += 1
        total += length
        removed += cutoff
    require(total == info["total_frames"], "total_frames differs")
    require(info["total_videos"] == len(cameras) * count, "total_videos differs")
    require(len(list((dataset / "videos").rglob("*.mp4"))) == len(cameras) * count
            if cameras else not (dataset / "videos").exists(), "Actual video count differs")
    summary = read_json(dataset / "meta/cleanup_summary.json")
    require(summary["removed_frames"] == removed, "Cleanup total differs")
    require(summary["remaining_frames"] == total, "Cleanup remaining total differs")
    modality = read_json(dataset / "meta/modality.json")
    require(sorted(value["original_key"] for value in modality["video"].values()) == video_keys,
            "Modality video keys differ")
    require(modality["action"]["gripper"]["absolute"] is True, "Action gripper must be absolute")
    if setup["action"] == "delta-joint":
        joints = modality["action"]["joints"]
        require(joints["absolute"] is False and joints["unit"] == setup.get("angle_unit", "radian"),
                "Delta joint modality must describe relative angles")
    elif setup["action"] == "delta-ee":
        frame = "measured_tcp_at_t" if setup.get("delta_ee_frame", "tool") == "tool" else "robot_base"
        for name in ("translation", "rotation_rotvec"):
            part = modality["action"][name]
            require(part["absolute"] is False and part["frame"] == frame,
                    "Delta EE modality must describe a relative pose in its declared frame")
    return {"passed": True, "episodes": count, "rows": total, "videos_checked": checked_videos,
            "action_offset": setup["action_offset"], "state": setup["state"],
            "action": setup["action"], "cameras": [camera["name"] for camera in cameras],
            "checks": ["raw_values", "action_alignment", "independent_idle_cutoff", "indices",
                       "episode_stats", "source_alignment", "metadata"]
                      + (["videos"] if videos and cameras else [])}


__all__ = ["expected_from_raw", "validate_dataset"]
