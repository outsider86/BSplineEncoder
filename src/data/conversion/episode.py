"""Per-episode alignment, state/action construction, and leading-idle cropping.

Temporal semantics (identical to ``scripts/legacy_piper/preprocess_piper.py``):

* The target clock spans the overlap of the robot stream and the SELECTED
  cameras only, at ``run.fps``; every selected stream is resampled by nearest
  timestamp (ties choose the earlier sample) BEFORE any difference is taken.
* Images, state, and extra observations are at ``t``.  Absolute actions are
  at ``t + action_offset``.  Deltas are ``t -> t+1`` and carry the absolute
  measured gripper at ``t+1``.  Rows without a next target are dropped.
* Leading idle rows are removed per episode; every stream is cropped by the
  same cutoff and time/frame indices restart at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..config import CameraSpec, ExtraObservation, RunConfig, Setup
from ..transforms.cleanup import detect_cutoff, episode_metrics, motion_signals, motion_source
from ..transforms.spaces import (
    canonicalize_quaternions, next_step_ee_actions, next_step_joint_actions,
)
from .media import CameraStream, video_properties
from .raw import (
    EE_FIELDS, JOINT_FIELDS, field_rows, load_payload, nearest_indices, rows,
    validate_timestamps,
)


@dataclass(frozen=True)
class Episode:
    source_dir: Path
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray
    extras: dict[str, np.ndarray]
    cameras: dict[str, CameraStream]
    camera_indices: dict[str, np.ndarray]
    alignment_ms: dict[str, np.ndarray]
    robot_indices: np.ndarray
    source_time: np.ndarray

    @property
    def length(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass(frozen=True)
class Aligned:
    episode: Episode
    measured_joint: np.ndarray
    measured_ee: np.ndarray | None
    command: np.ndarray | None
    robot_indices: np.ndarray
    target_source_time: np.ndarray


def _camera_stream(path: Path, payload: dict, camera: CameraSpec, robot_time: np.ndarray,
                   count: int) -> CameraStream:
    video = path / camera.file
    width, height, frames, source_fps = video_properties(video)
    raw_times = payload.get("camera_timestamps")
    if raw_times is None:
        if frames != count:
            raise ValueError(f"{video}: legacy frame count {frames} != robot samples {count}")
        times = robot_time.copy()
    else:
        if camera.timestamps_key not in raw_times:
            raise ValueError(f"{path}: camera_timestamps has no {camera.timestamps_key!r}")
        times = validate_timestamps(raw_times[camera.timestamps_key], str(video), frames)
    acquisition = payload.get("camera_frame_source_timestamps")
    if acquisition is None:
        capture = times
    else:
        if camera.timestamps_key not in acquisition:
            raise ValueError(f"{path}: camera_frame_source_timestamps has no "
                             f"{camera.timestamps_key!r}")
        capture = validate_timestamps(acquisition[camera.timestamps_key],
                                      f"{video}: acquisition", frames)
    return CameraStream(camera.name, video, times, capture, width, height, source_fps)


def load_episode(path: Path, setup: Setup, run: RunConfig,
                 cameras: tuple[CameraSpec, ...] = (),
                 extras: tuple[ExtraObservation, ...] = ()) -> Aligned:
    """Resample selected streams first, then select state[t] and action[t(+1)]."""

    setup.check()
    path = Path(path)
    payload = load_payload(path)
    observations = payload["observations"]
    count = len(observations)
    if count < 2:
        raise ValueError(f"{path}: at least two robot samples are required")
    robot_time = validate_timestamps(payload["timestamps"], str(path), count)
    joint = rows(observations, JOINT_FIELDS, "observations")
    ee = None
    if setup.needs_ee or run.cleanup_motion == "measured-ee":
        ee = rows(observations, EE_FIELDS, "observations")
        if np.any(np.linalg.norm(ee[:, 3:7], axis=1) < 1e-8):
            raise ValueError(f"{path}: near-zero measured EE quaternion")
        if setup.quaternion == "continuous":
            ee[:, 3:7] = canonicalize_quaternions(ee[:, 3:7], str(path))
    command = previous = None
    if setup.needs_command or run.cleanup_motion == "command-joint":
        commands = payload.get("actions", [])
        if len(commands) != count:
            raise ValueError(f"{path}: {len(commands)} commands vs {count} observations")
        command = rows(commands, JOINT_FIELDS, "actions")
        previous = np.concatenate([command[:1], command[:-1]])
    extra_values = {extra.key: field_rows(observations, extra.field, "observations")
                    for extra in extras}

    streams = {camera.name: _camera_stream(path, payload, camera, robot_time, count)
               for camera in cameras}
    start = max([robot_time[0], *(stream.timestamps[0] for stream in streams.values())])
    end = min([robot_time[-1], *(stream.timestamps[-1] for stream in streams.values())])
    target_count = int(np.floor((end - start) * run.fps + 1e-9)) + 1 if end >= start else 0
    if target_count < 2:
        raise ValueError(f"{path}: streams do not share at least two target samples")
    target = start + np.arange(target_count, dtype=np.float64) / run.fps
    selected = nearest_indices(robot_time, target)
    joint = joint[selected]
    if ee is not None:
        ee = ee[selected]
        if setup.quaternion == "continuous":
            # Historical behaviour: canonicalize on the raw clock and again on
            # the target clock so skipped/repeated rows cannot flip signs.
            ee[:, 3:7] = canonicalize_quaternions(ee[:, 3:7], f"{path}: resampled EE")
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
        action = next_step_ee_actions(ee, setup.delta_ee_frame)
    elif setup.action == "delta-joint":
        action = next_step_joint_actions(joint)
    else:
        trajectory = {"joint": joint, "command": command, "abs-ee": ee}[setup.action]
        action = trajectory[setup.action_offset:]
    camera_indices, alignment = {}, {
        "robot": np.abs(robot_time[selected[:stop]] - target[:stop]) * 1000.0
    }
    for name, stream in streams.items():
        indices = nearest_indices(stream.timestamps, target[:stop])
        camera_indices[name] = indices
        alignment[name] = np.abs(stream.capture_timestamps[indices] - target[:stop]) * 1000.0
    episode = Episode(
        path, np.arange(stop, dtype=np.float32) / run.fps, state[:stop], action,
        {key: values[selected[:stop]] for key, values in extra_values.items()},
        streams, camera_indices, alignment, selected[:stop], target[:stop],
    )
    return Aligned(episode, joint, ee, command, selected, target)


def clip_episode(aligned: Aligned, setup: Setup, run: RunConfig, index: int):
    """Crop the leading idle prefix from every stream with one shared cutoff."""

    episode = aligned.episode
    config = run.detector()
    source = motion_source(setup, run)
    length = episode.length
    if source == "measured-ee":
        if aligned.measured_ee is None:
            raise ValueError("measured-ee cleanup requires measured EE feedback")
        signals = motion_signals(aligned.measured_ee[:-1], next_step_ee_actions(aligned.measured_ee),
                                 config, "observation_ee", "observation_delta_ee")
    elif source == "measured-delta-joint":
        signals = motion_signals(aligned.measured_joint[:-1],
                                 next_step_joint_actions(aligned.measured_joint),
                                 config, "state", "observation_delta_joint")
    else:
        if source == "command-joint" and aligned.command is None:
            raise ValueError("command-joint cleanup requires recorded commands")
        trajectory = (aligned.command[setup.action_offset:] if source == "command-joint"
                      else aligned.measured_joint[:length])
        signals = motion_signals(trajectory, trajectory, config, "state",
                                 "action" if source == "command-joint" else "observation")
    cutoff = detect_cutoff(signals, length, config) if run.cleanup else 0
    metrics = episode_metrics(index, length, cutoff, run.fps, signals)
    metrics["source_episode"] = str(episode.source_dir.resolve())
    metrics["cleanup_motion_source"] = source
    cropped = replace(
        episode,
        timestamps=np.arange(length - cutoff, dtype=np.float32) / run.fps,
        state=episode.state[cutoff:], action=episode.action[cutoff:],
        extras={key: values[cutoff:] for key, values in episode.extras.items()},
        camera_indices={key: values[cutoff:] for key, values in episode.camera_indices.items()},
        alignment_ms={key: values[cutoff:] for key, values in episode.alignment_ms.items()},
        robot_indices=episode.robot_indices[cutoff:], source_time=episode.source_time[cutoff:],
    )
    return cropped, metrics


__all__ = ["Aligned", "Episode", "clip_episode", "load_episode"]
