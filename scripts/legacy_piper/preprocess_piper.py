#!/usr/bin/env python3
"""Configurable Piper RawData -> LeRobot v2.1 preprocessing.

Examples (run in the starVLA environment):
  python Data/RawData/preprocess_piper.py --list-setups
  python Data/RawData/preprocess_piper.py --setup state-abs-ee --dry-run
  python Data/RawData/preprocess_piper.py --setup state-abs-ee --setup state-abs-joint
  python Data/RawData/preprocess_piper.py --setup state-delta-joint --setup state-delta-ee --jobs 3
  python Data/RawData/preprocess_piper.py --state joint-last-command --action abs-ee
  python Data/RawData/preprocess_piper.py --all-setups --jobs 3

Images and state are always at t. Absolute actions default to t+1; historical
presets explicitly select their original offset. Delta joint and delta EE always
mean t->t+1, with absolute measured gripper[t+1]. Joint deltas are q[t+1]-q[t]
in radians; EE deltas are translation/rotvec in the current measured TCP frame.
Command history defaults to the previous
RAW robot sample, matching the original converter; --last-command-clock target
selects the previous resampled sample instead. Leading idle removal and raw-data
validation are enabled by default. Videos are encoded once, after cropping.
Existing datasets are replaced only with --overwrite. Original converters are
still available with their existing interfaces.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from convert_piper_to_lerobot import (
    CAMERAS, CameraStream, Episode, _features, _nearest_indices,
    _validate_timestamps, _video_properties, _write_episode_parquet,
    _write_json, _write_resampled_video,
)
from convert_piper_to_lerobot_delta_ee import _canonicalize_quaternions, _next_step_actions
from clean_static_frames_LeRobot import (
    CleanupConfig, MotionSignals, _detect_cutoff, _episode_metrics,
    _ffmpeg_path, _motion_signals, _summary, _write_metrics_csv,
)


DATA_ROOT = Path(__file__).resolve().parent.parent
TASKS = {
    "classify_blocks": "Classify the blocks.",
    "hanging_mug": "Hang the mug.",
    "stacking_cup": "Stack the cups.",
}
STATES = ("joint", "joint-last-command", "last-command", "ee", "ee-last-command")
ACTIONS = ("joint", "command", "abs-ee", "delta-joint", "delta-ee")
STATE_SOURCE = {
    "joint": "state", "joint-last-command": "state_last_command",
    "last-command": "last_command", "ee": "observation_ee",
    "ee-last-command": "observation_ee_last_command",
}
ACTION_SOURCE = {
    "joint": "observation", "command": "action",
    "abs-ee": "observation_ee", "delta-joint": "observation_delta_joint",
    "delta-ee": "observation_delta_ee",
}


@dataclass(frozen=True)
class Setup:
    state: str
    action: str
    action_offset: int = 1
    quaternion: str = "raw"
    last_command_clock: str = "raw"

    def check(self) -> None:
        if self.state not in STATES or self.action not in ACTIONS:
            raise ValueError(f"Unknown state/action: {self.state}/{self.action}")
        if self.action_offset not in (0, 1):
            raise ValueError("action-offset must be 0 or 1")
        if self.action.startswith("delta-") and self.action_offset != 1:
            raise ValueError(f"{self.action} requires --action-offset 1: measured state[t] -> state[t+1]")
        if self.quaternion not in ("raw", "continuous"):
            raise ValueError("quaternion must be raw or continuous")
        if self.last_command_clock not in ("raw", "target"):
            raise ValueError("last-command-clock must be raw or target")


# The preset paths reproduce the dataset families used in this project.
PRESETS: dict[str, tuple[Setup, str]] = {
    "state-abs-ee": (Setup("joint", "abs-ee"), "StateAbsEE"),
    "state-abs-joint": (Setup("joint", "joint"), "StateAbsJoint"),
    "state-delta-joint": (Setup("joint", "delta-joint"), "StateDeltaJoint"),
    "state-delta-ee": (Setup("joint", "delta-ee", quaternion="continuous"), "StateDeltaEE"),
    "ee-delta-ee": (Setup("ee", "delta-ee", quaternion="continuous"), "ActionEE/State_EE"),
    "joint-command": (Setup("joint", "command", 0), "ActionCommand/State_Joint"),
    "joint-last-command-command": (Setup("joint-last-command", "command", 0), "ActionCommand/State_LastCommand_Joint"),
    "last-command-command": (Setup("last-command", "command", 0), "ActionCommand/LastCommand_Joint"),
    "joint-joint": (Setup("joint", "joint", 0), "ActionJoint/State_Joint"),
    "joint-last-command-joint": (Setup("joint-last-command", "joint", 0), "ActionJoint/State_LastCommand_Joint"),
    "last-command-joint": (Setup("last-command", "joint", 0), "ActionJoint/LastCommand_Joint"),
    "processed": (Setup("joint-last-command", "command", 0), "Processed"),
}


@dataclass(frozen=True)
class RunConfig:
    fps: float = 30.0
    cleanup: bool = True
    cleanup_motion: str = "auto"
    arm_threshold: float = 0.005
    gripper_threshold: float = 0.01
    translation_threshold: float = 0.0005
    rotation_threshold: float = 0.005
    confirmation_seconds: float = 0.4
    confirmation_fraction: float = 0.5
    preroll_seconds: float = 0.0
    max_scan_seconds: float = 10.0
    min_remaining_seconds: float = 2.0

    def detector(self) -> CleanupConfig:
        if self.cleanup_motion not in ("auto", "measured-joint", "measured-delta-joint", "command-joint", "measured-ee"):
            raise ValueError("Unknown cleanup motion source")
        values = (self.fps, self.arm_threshold, self.gripper_threshold,
                  self.translation_threshold, self.rotation_threshold,
                  self.confirmation_seconds, self.max_scan_seconds, self.min_remaining_seconds)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("FPS, thresholds and durations must be finite and positive")
        if not np.isfinite(self.preroll_seconds) or self.preroll_seconds < 0:
            raise ValueError("preroll-seconds must be finite and nonnegative")
        if not 0 < self.confirmation_fraction <= 1:
            raise ValueError("confirmation-fraction must be in (0, 1]")
        window = max(1, round(self.confirmation_seconds * self.fps))
        return CleanupConfig(
            self.arm_threshold, self.gripper_threshold,
            self.translation_threshold, self.rotation_threshold,
            window, max(1, round(window * self.confirmation_fraction)),
            round(self.preroll_seconds * self.fps), round(self.max_scan_seconds * self.fps),
            round(self.min_remaining_seconds * self.fps),
        )


@dataclass(frozen=True)
class Aligned:
    episode: Episode
    measured_joint: np.ndarray
    measured_ee: np.ndarray | None
    command: np.ndarray | None
    robot_indices: np.ndarray
    target_source_time: np.ndarray


def _rows(rows: list[Any], keys: tuple[tuple[str, int], ...], label: str) -> np.ndarray:
    vectors = []
    for index, row in enumerate(rows):
        parts = []
        for key, dim in keys:
            if key not in row:
                raise ValueError(f"{label}[{index}]: missing {key!r}")
            value = np.asarray(row[key], dtype=np.float32).reshape(-1)
            if value.shape != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"{label}[{index}].{key}: expected finite ({dim},), got {value}")
            parts.append(value)
        vectors.append(np.concatenate(parts))
    return np.stack(vectors)


def _next_joint_actions(measured_joint: np.ndarray) -> np.ndarray:
    """Six measured joint increments in radians and next absolute gripper."""
    if measured_joint.ndim != 2 or measured_joint.shape[1] != 7 or len(measured_joint) < 2:
        raise ValueError("Measured joints must have shape (N, 7) with N >= 2")
    return np.concatenate(
        [np.diff(measured_joint[:, :6], axis=0), measured_joint[1:, 6:7]], axis=1
    ).astype(np.float32)


def load_episode(path: Path, setup: Setup, run: RunConfig) -> Aligned:
    """Resample first, then select current observation and current/next action."""
    setup.check()
    with (path / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    observations = payload["observations"]
    count = len(observations)
    if count < 2:
        raise ValueError(f"{path}: at least two robot samples are required")
    robot_time = _validate_timestamps(payload["timestamps"], str(path), count)
    joint = _rows(observations, (("arm_joint_position", 6), ("gripper_pos", 1)), "observations")
    need_ee = (setup.state.startswith("ee") or "ee" in setup.action
               or run.cleanup_motion == "measured-ee")
    ee = None
    if need_ee:
        ee = _rows(observations, (("arm_pos", 3), ("arm_quat", 4), ("gripper_pos", 1)), "observations")
        if np.any(np.linalg.norm(ee[:, 3:7], axis=1) < 1e-8):
            raise ValueError(f"{path}: near-zero measured EE quaternion")
        if setup.quaternion == "continuous":
            ee[:, 3:7] = _canonicalize_quaternions(ee[:, 3:7], str(path))
    need_command = ("command" in setup.state or setup.action == "command"
                    or run.cleanup_motion == "command-joint")
    command = previous = None
    if need_command:
        commands = payload.get("actions", [])
        if len(commands) != count:
            raise ValueError(f"{path}: {len(commands)} commands vs {count} observations")
        command = _rows(commands, (("arm_joint_position", 6), ("gripper_pos", 1)), "actions")
        previous = np.concatenate([command[:1], command[:-1]])

    cameras = {}
    for camera, filename in CAMERAS.items():
        video = path / filename
        width, height, frames, source_fps = _video_properties(video)
        raw_times = payload.get("camera_timestamps")
        if raw_times is None:
            if frames != count:
                raise ValueError(f"{video}: legacy frame count {frames} != robot samples {count}")
            times = robot_time.copy()
        else:
            times = _validate_timestamps(raw_times[camera], str(video), frames)
        acquisition = payload.get("camera_frame_source_timestamps")
        capture_time = times if acquisition is None else _validate_timestamps(
            acquisition[camera], f"{video}: acquisition", frames
        )
        cameras[camera] = CameraStream(video, times, capture_time, width, height, source_fps)
    start = max(robot_time[0], *(camera.timestamps[0] for camera in cameras.values()))
    end = min(robot_time[-1], *(camera.timestamps[-1] for camera in cameras.values()))
    target_count = int(np.floor((end - start) * run.fps + 1e-9)) + 1
    if target_count < 2:
        raise ValueError(f"{path}: streams do not share at least two target samples")
    target = start + np.arange(target_count, dtype=np.float64) / run.fps
    selected = _nearest_indices(robot_time, target)
    joint = joint[selected]
    ee = None if ee is None else ee[selected]
    if ee is not None and setup.quaternion == "continuous":
        # Match the historical delta-EE converter: normalize/canonicalize on
        # the raw clock and again on the selected target clock.
        ee[:, 3:7] = _canonicalize_quaternions(ee[:, 3:7], f"{path}: resampled EE")
    if command is not None:
        command = command[selected]
        previous = (previous[selected] if setup.last_command_clock == "raw"
                    else np.concatenate([command[:1], command[:-1]]))
    state = {
        "joint": lambda: joint,
        "joint-last-command": lambda: np.concatenate([joint, previous], axis=1),
        "last-command": lambda: previous,
        "ee": lambda: ee,
        "ee-last-command": lambda: np.concatenate([ee, previous], axis=1),
    }[setup.state]()
    stop = target_count - setup.action_offset
    if setup.action == "delta-ee":
        action = _next_step_actions(ee)
    elif setup.action == "delta-joint":
        action = _next_joint_actions(joint)
    else:
        trajectory = {"joint": joint, "command": command, "abs-ee": ee}[setup.action]
        action = trajectory[setup.action_offset:]
    camera_indices = {}
    alignment = {"robot": np.abs(robot_time[selected[:stop]] - target[:stop]) * 1000.0}
    for name, camera in cameras.items():
        indices = _nearest_indices(camera.timestamps, target[:stop])
        camera_indices[name] = indices
        alignment[name] = np.abs(camera.capture_timestamps[indices] - target[:stop]) * 1000.0
    episode = Episode(
        path, np.arange(stop, dtype=np.float32) / run.fps, state[:stop], action,
        cameras, camera_indices, alignment,
    )
    return Aligned(episode, joint, ee, command, selected, target)


def motion_source(setup: Setup, run: RunConfig) -> str:
    if run.cleanup_motion != "auto":
        return run.cleanup_motion
    return {"command": "command-joint", "delta-ee": "measured-ee",
            "delta-joint": "measured-delta-joint"}.get(setup.action, "measured-joint")


def clip_episode(aligned: Aligned, setup: Setup, run: RunConfig, index: int):
    episode = aligned.episode
    config = run.detector()
    source = motion_source(setup, run)
    length = episode.length
    if source == "measured-ee":
        delta = _next_step_actions(aligned.measured_ee)
        current = aligned.measured_ee[:-1]
        # These signals start at t and may include t->t+1 on the last row.
        signals = _motion_signals(current, delta, config, "observation_ee", "observation_delta_ee")
    elif source == "measured-delta-joint":
        signals = _motion_signals(
            aligned.measured_joint[:-1], _next_joint_actions(aligned.measured_joint),
            config, "state", "observation_delta_joint",
        )
    else:
        trajectory = (aligned.command[setup.action_offset:] if source == "command-joint"
                      else aligned.measured_joint[:length])
        signals = _motion_signals(
            trajectory, trajectory, config, "state",
            "action" if source == "command-joint" else "observation",
        )
    cutoff = _detect_cutoff(signals, length, config) if run.cleanup else 0
    metrics = _episode_metrics(index, length, cutoff, run.fps, signals)
    metrics["source_episode"] = str(episode.source_dir.resolve())
    metrics["cleanup_motion_source"] = source
    cropped = replace(
        episode, timestamps=np.arange(length - cutoff, dtype=np.float32) / run.fps,
        state=episode.state[cutoff:], action=episode.action[cutoff:],
        camera_indices={key: values[cutoff:] for key, values in episode.camera_indices.items()},
        alignment_ms={key: values[cutoff:] for key, values in episode.alignment_ms.items()},
    )
    return cropped, metrics


def modality(setup: Setup) -> tuple[dict, dict, str, str]:
    def block(key, start, end, **extra):
        return {"original_key": key, "start": start, "end": end,
                "absolute": True, "dtype": "float32", **extra}
    state = {}
    state_layout = []
    prefix = 0
    if setup.state.startswith("joint"):
        state.update(measured_joints=block("observation.state", 0, 6, unit="radian"),
                     measured_gripper=block("observation.state", 6, 7, range=[0, 1]))
        state_layout.append("measured joints/gripper[t]")
        prefix = 7
    elif setup.state.startswith("ee"):
        state.update(measured_tcp_position=block("observation.state", 0, 3, unit="meter", frame="robot_base"),
                     measured_tcp_quaternion_xyzw=block("observation.state", 3, 7, quaternion_order="xyzw"),
                     measured_gripper=block("observation.state", 7, 8, range=[0, 1]))
        state_layout.append("measured EE xyz/quaternion xyzw/gripper[t]")
        prefix = 8
    if "command" in setup.state:
        state.update(previous_command_joints=block("observation.state", prefix, prefix + 6, unit="radian"),
                     previous_command_gripper=block("observation.state", prefix + 6, prefix + 7, range=[0, 1]))
        state_layout.append(f"previous command[{setup.last_command_clock} clock; first repeats first]")
    time = "t+1" if setup.action_offset else "t"
    if setup.action in ("joint", "command"):
        action = {"joints": block("action", 0, 6, unit="radian"),
                  "gripper": block("action", 6, 7, range=[0, 1])}
        action_layout = f"{'measured' if setup.action == 'joint' else 'command'} joints/gripper[{time}]"
    elif setup.action == "delta-joint":
        action = {"joints": block("action", 0, 6, absolute=False, unit="radian", frame="joint_space"),
                  "gripper": block("action", 6, 7, range=[0, 1])}
        action_layout = "measured joint delta q[t+1]-q[t] + absolute measured gripper[t+1]"
    elif setup.action == "abs-ee":
        action = {"position": block("action", 0, 3, unit="meter", frame="robot_base"),
                  "quaternion": block("action", 3, 7, quaternion_order="xyzw"),
                  "gripper": block("action", 7, 8, range=[0, 1])}
        action_layout = f"measured absolute EE xyz/quaternion xyzw/gripper[{time}]"
    else:
        action = {"translation": block("action", 0, 3, absolute=False, unit="meter", frame="measured_tcp_at_t"),
                  "rotation_rotvec": block("action", 3, 6, absolute=False, unit="radian", frame="measured_tcp_at_t"),
                  "gripper": block("action", 6, 7, range=[0, 1])}
        action_layout = "measured local EE delta[t->t+1] + absolute measured gripper[t+1]"
    return state, action, " + ".join(state_layout), action_layout


def convert(raw_task: Path, destination: Path, task: str, setup: Setup, run: RunConfig,
            preset: str, overwrite: bool = False, validate: bool = True,
            check_videos: bool = True, limit: int | None = None) -> Path:
    setup.check()
    run.detector()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {destination}; use --overwrite to replace")
    sources = sorted({path.parent for path in raw_task.rglob("data.pkl")})
    if limit is not None:
        if limit < 1:
            raise ValueError("limit-episodes must be positive")
        sources = sources[:limit]
    if not sources:
        raise ValueError(f"No raw episodes under {raw_task}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}_", dir=destination.parent))
    try:
        records, metrics, selection = [], [], []
        total = 0
        reference = None
        for index, path in enumerate(sources):
            aligned = load_episode(path, setup, run)
            episode, metric = clip_episode(aligned, setup, run, index)
            if reference is None:
                reference = episode
            for camera in CAMERAS:
                actual = episode.cameras[camera]
                expected = reference.cameras[camera]
                if (actual.width, actual.height) != (expected.width, expected.height):
                    raise ValueError(f"{path}: {camera} resolution changed")
            _write_episode_parquet(staging / f"data/chunk-000/episode_{index:06d}.parquet", episode, index, total)
            for camera in CAMERAS:
                _write_resampled_video(episode.cameras[camera], episode.camera_indices[camera],
                    staging / f"videos/chunk-000/observation.images.{camera}/episode_{index:06d}.mp4", run.fps,
                    ffmpeg=os.environ.get("IMAGEIO_FFMPEG_EXE"))
            records.append({"episode_index": index, "tasks": [task], "length": episode.length})
            metrics.append(metric)
            selection.append({"episode_index": index, "source_episode": str(path.resolve()),
                              "resampled_frames": len(aligned.robot_indices),
                              "dropped_tail_frames": setup.action_offset,
                              "removed_prefix_frames": metric["removed_frames"]})
            total += episode.length
            print(f"[{preset} {raw_task.name} {index+1}/{len(sources)}] {path.name}: "
                  f"{episode.length} frames, removed {metric['removed_frames']}", flush=True)
        state_mod, action_mod, state_layout, action_layout = modality(setup)
        info = {
            "codebase_version": "v2.1", "robot_type": "piper", "fps": run.fps,
            "total_episodes": len(sources), "total_frames": total, "total_tasks": 1,
            "total_videos": len(sources) * len(CAMERAS), "total_chunks": 1, "chunks_size": 1000,
            "splits": {"train": f"0:{len(sources)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": _features(reference.cameras, run.fps, reference.state.shape[1], reference.action.shape[1]),
            "state_source": STATE_SOURCE[setup.state], "action_source": ACTION_SOURCE[setup.action],
            "state_layout": state_layout, "action_layout": action_layout,
            "action_timestep_offset": setup.action_offset,
            "preprocessing": {"version": 1, "preset": preset, "setup": asdict(setup),
                              "run": asdict(run), "cleanup_motion_source": motion_source(setup, run),
                              "raw_root": str(raw_task.resolve()), "limit_episodes": limit},
        }
        summary = _summary(raw_task, destination.name, run.fps, metrics, run.detector())
        summary["method"] = "first sustained motion" if run.cleanup else "disabled"
        summary["motion_source"] = motion_source(setup, run)
        info["cleanup"] = {"enabled": run.cleanup, "method": summary["method"],
                           "removed_frames": summary["removed_frames"], "config": summary["config"],
                           "motion_source": summary["motion_source"]}
        _write_json(staging / "meta/info.json", info)
        _write_json(staging / "meta/modality.json", {
            "video": {camera: {"original_key": f"observation.images.{camera}"} for camera in CAMERAS},
            "state": state_mod, "action": action_mod,
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        })
        _write_json(staging / "meta/embodiment.json", {
            "robot_name": "Piper", "robot_type": "piper", "embodiment_tag": "new_embodiment",
            "record_frequency": run.fps, "body_controller_frequency": run.fps,
            "hand_controller_frequency": run.fps, "state_source": info["state_source"],
            "action_source": info["action_source"], "state_layout": state_layout,
            "action_layout": action_layout, "action_timestep_offset": setup.action_offset,
        })
        _write_json(staging / "meta/cleanup_summary.json", summary)
        _write_metrics_csv(staging / "meta/cleanup_metrics.csv", metrics)
        for filename, rows in (("episodes.jsonl", records), ("tasks.jsonl", [{"task_index": 0, "task": task}]),
                               ("source_episodes.jsonl", selection)):
            (staging / "meta" / filename).write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        if validate:
            from validate_piper_preprocessing import validate_dataset

            result = validate_dataset(staging, check_videos)
            _write_json(staging / "meta/validation.json", result)
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Saved {destination}: {len(sources)} episodes / {total} frames", flush=True)
    return destination


def resolve_ffmpeg(explicit: Path | None) -> str:
    try:
        return _ffmpeg_path(explicit)
    except RuntimeError:
        # Conda's base environment may have imageio-ffmpeg even if starVLA does not.
        base = Path(sys.prefix).parent.parent
        candidates = sorted(base.glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*"))
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        raise


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list-setups", action="store_true")
    p.add_argument("--setup", choices=tuple(PRESETS), action="append", help="Repeat to run multiple presets")
    p.add_argument("--all-setups", action="store_true")
    p.add_argument("--state", choices=STATES, help="Override preset state or define a custom setup")
    p.add_argument("--action", choices=ACTIONS, help="Override preset action or define a custom setup")
    p.add_argument("--action-offset", type=int, choices=(0, 1), help="0=current, 1=next target; custom defaults to 1")
    p.add_argument("--quaternion", choices=("raw", "continuous"), help="Raw feedback or normalized quaternions with continuous signs")
    p.add_argument("--last-command-clock", choices=("raw", "target"), help="Previous raw robot sample or previous target-FPS sample")
    p.add_argument("--raw-root", type=Path, default=DATA_ROOT / "RawData")
    p.add_argument("--output-root", type=Path, default=DATA_ROOT, help="Parent of preset families; custom options use Custom/... beneath this")
    p.add_argument("--tasks", nargs="+", help="Raw task subfolders; default: discover all folders containing data.pkl")
    p.add_argument("--task-description", help="Custom language instruction (one task only)")
    p.add_argument("--target-fps", type=float, default=30.0)
    p.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cleanup-motion", choices=("auto", "measured-joint", "measured-delta-joint", "command-joint", "measured-ee"), default="auto")
    p.add_argument("--arm-delta-threshold", type=float, default=0.005)
    p.add_argument("--gripper-delta-threshold", type=float, default=0.01)
    p.add_argument("--ee-translation-delta-threshold", type=float, default=0.0005)
    p.add_argument("--ee-rotation-delta-threshold", type=float, default=0.005)
    p.add_argument("--confirmation-window-seconds", type=float, default=0.4)
    p.add_argument("--confirmation-min-fraction", type=float, default=0.5)
    p.add_argument("--preroll-seconds", type=float, default=0.0)
    p.add_argument("--max-scan-seconds", type=float, default=10.0)
    p.add_argument("--min-remaining-seconds", type=float, default=2.0)
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-video-validation", action="store_true")
    p.add_argument("--limit-episodes", type=int, help="Convert only the first N episodes per task for a smoke test")
    p.add_argument("--jobs", type=int, default=1, help="Concurrent task/setup jobs")
    p.add_argument("--ffmpeg", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print resolved configuration and output paths without writing")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.list_setups:
        for name, (setup, folder) in PRESETS.items():
            print(f"{name:30} state={setup.state:19} action={setup.action:8} offset={setup.action_offset} -> {folder}")
        return 0
    if args.all_setups and args.setup:
        raise ValueError("Choose --all-setups or --setup, not both")
    names = list(PRESETS) if args.all_setups else args.setup or ["custom"]
    if names == ["custom"] and (args.state is None or args.action is None):
        raise ValueError("Select --setup, --all-setups, or both --state and --action")
    if args.jobs < 1 or (args.limit_episodes is not None and args.limit_episodes < 1):
        raise ValueError("jobs and limit-episodes must be positive")
    run = RunConfig(
        args.target_fps, args.cleanup, args.cleanup_motion,
        args.arm_delta_threshold, args.gripper_delta_threshold,
        args.ee_translation_delta_threshold, args.ee_rotation_delta_threshold,
        args.confirmation_window_seconds, args.confirmation_min_fraction,
        args.preroll_seconds, args.max_scan_seconds, args.min_remaining_seconds,
    )
    run.detector()
    raw_root = args.raw_root.expanduser().resolve()
    tasks = args.tasks or sorted(p.name for p in raw_root.iterdir()
                                if p.is_dir() and any(p.rglob("data.pkl")))
    if not tasks:
        raise ValueError(f"No tasks containing raw episodes under {raw_root}")
    if len(set(tasks)) != len(tasks):
        raise ValueError("Duplicate tasks")
    if args.task_description and len(tasks) != 1:
        raise ValueError("task-description requires exactly one task")
    jobs = []
    for name in names:
        base, family = PRESETS.get(name, (Setup(args.state, args.action), "Custom"))
        changes = {field: value for field, value in (
            ("state", args.state), ("action", args.action), ("action_offset", args.action_offset),
            ("quaternion", args.quaternion), ("last_command_clock", args.last_command_clock),
        ) if value is not None}
        setup = replace(base, **changes)
        setup.check()
        if name == "custom":
            family = (f"Custom/{setup.state}__{setup.action}__offset{setup.action_offset}"
                      f"__quat-{setup.quaternion}__last-{setup.last_command_clock}"
                      f"__motion-{motion_source(setup, run)}")
        for task_name in tasks:
            if Path(task_name).name != task_name or task_name in (".", ".."):
                raise ValueError("Each task must be one raw subfolder name")
            raw_task = raw_root / task_name
            instruction = args.task_description or TASKS.get(task_name)
            if not instruction:
                raise ValueError(f"Unknown task {task_name}: provide --tasks {task_name} --task-description")
            if not raw_task.is_dir() or not any(raw_task.rglob("data.pkl")):
                raise ValueError(f"No raw episodes in {raw_task}")
            dataset_name = f"{task_name}_{run.fps:g}hz" + ("_cleanup" if run.cleanup else "")
            output = args.output_root.expanduser().resolve() / family / dataset_name
            jobs.append((raw_task, output, instruction, setup, run, name))
    outputs = [job[1] for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Duplicate output paths in this run")
    for raw_task, output, instruction, setup, run, name in jobs:
        print(json.dumps({"preset": name, "raw": str(raw_task), "output": str(output),
                          "setup": asdict(setup), "run": asdict(run), "task": instruction}, indent=2))
        if output.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
        # A destination cannot contain or replace the raw episode tree.
        if output == raw_root or output in raw_root.parents or raw_root in output.parents:
            raise ValueError("Output must be outside RawData")
    if args.dry_run:
        return 0
    os.environ["IMAGEIO_FFMPEG_EXE"] = resolve_ffmpeg(args.ffmpeg)

    def execute(job):
        return convert(*job, overwrite=args.overwrite, validate=args.validate,
                       check_videos=not args.skip_video_validation, limit=args.limit_episodes)
    if args.jobs == 1:
        for job in jobs:
            execute(job)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            list(executor.map(execute, jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
