"""Leading-idle detection, ported from ``scripts/legacy_piper/clean_static_frames_LeRobot.py``.

Only the leading idle prefix of each episode is removed.  Internal pauses are
never touched and separate segments are never joined: the retained rows are
always one contiguous suffix ``[cutoff:]`` of the aligned episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # config imports this module; no runtime cycle
    from ..config import RunConfig, Setup


@dataclass(frozen=True)
class CleanupConfig:
    arm_delta_threshold: float
    gripper_delta_threshold: float
    ee_translation_delta_threshold: float
    ee_rotation_delta_threshold: float
    confirmation_window_frames: int
    confirmation_min_moving_frames: int
    preroll_frames: int
    max_scan_frames: int
    min_remaining_frames: int


@dataclass(frozen=True)
class MotionSignals:
    command_arm_delta: np.ndarray
    command_gripper_delta: np.ndarray
    measured_arm_delta: np.ndarray
    measured_gripper_delta: np.ndarray
    ee_translation_delta: np.ndarray
    ee_rotation_delta: np.ndarray
    moving: np.ndarray
    command_available: bool
    measured_available: bool
    motion_frame_offset: int


def motion_signals(
    state: np.ndarray,
    action: np.ndarray,
    config: CleanupConfig,
    state_source: str | None = None,
    action_source: str | None = None,
    action_timestep_offset: int = 0,
) -> MotionSignals:
    """Per-transition motion flags.

    Joint trajectories mark transition ``i -> i+1`` and use
    ``motion_frame_offset=1`` so the first changed row is kept.  Delta rows
    already describe motion starting at row ``i`` and use offset 0.
    """

    if state.ndim != 2 or state.shape[1] < 7:
        raise ValueError(f"observation.state must have at least 7 values, got {state.shape}")
    expected_action_dim = 8 if action_source == "observation_ee" else 7
    if action.ndim != 2 or action.shape[1] != expected_action_dim:
        raise ValueError(
            f"action must have {expected_action_dim} values for "
            f"action_source={action_source!r}, got {action.shape}"
        )
    if len(state) != len(action):
        raise ValueError(f"state/action lengths differ: {len(state)} != {len(action)}")

    if action_source == "observation_delta_joint":
        if state_source not in ("state", "state_last_command"):
            raise ValueError(
                "observation_delta_joint cleanup requires current measured joint/gripper state"
            )
        measured_arm = np.linalg.norm(action[:, :6], axis=1)
        measured_gripper = np.abs(action[:, 6] - state[:, 6])
        moving = ((measured_arm > config.arm_delta_threshold)
                  | (measured_gripper > config.gripper_delta_threshold))
        zeros = np.zeros(len(action), dtype=np.float32)
        return MotionSignals(zeros, zeros, measured_arm, measured_gripper, zeros, zeros,
                             moving, False, True, 0)

    if action_source == "observation_delta_ee":
        gripper_index = 6 if state_source in ("state", "state_last_command") else 7
        if state_source == "last_command" or state.shape[1] <= gripper_index:
            raise ValueError(
                "observation_delta_ee cleanup requires a current measured gripper "
                f"in observation.state, got state_source={state_source!r}, shape={state.shape}"
            )
        translation = np.linalg.norm(action[:, :3], axis=1)
        rotation = np.linalg.norm(action[:, 3:6], axis=1)
        measured_gripper = np.abs(action[:, 6] - state[:, gripper_index])
        moving = ((translation > config.ee_translation_delta_threshold)
                  | (rotation > config.ee_rotation_delta_threshold)
                  | (measured_gripper > config.gripper_delta_threshold))
        zeros = np.zeros(len(action), dtype=np.float32)
        return MotionSignals(zeros, zeros, zeros, measured_gripper, translation, rotation,
                             moving, False, True, 0)

    measured: np.ndarray | None = None
    command: np.ndarray | None = None
    if state_source == "state_last_command":
        if state.shape[1] != 14:
            raise ValueError(f"state_last_command requires 14 values, got {state.shape}")
        measured, command = state[:, :7], state[:, 7:14]
    elif state_source == "state":
        measured = state[:, :7]
    elif state_source == "last_command":
        command = state[:, :7]
    elif state_source is None:
        measured = state[:, :7]
        if state.shape[1] >= 14:
            command = state[:, 7:14]

    if action_source == "observation" and action_timestep_offset == 0:
        measured = action
    elif action_source == "observation_ee":
        if measured is None:
            raise ValueError("observation_ee cleanup requires measured joint state")
    elif action_source in {"action", "command"}:
        command = action
    elif action_source is None and command is None:
        command = action

    transitions = len(state) - 1
    if command is None:
        command_arm = np.zeros(transitions, dtype=np.float32)
        command_gripper = np.zeros(transitions, dtype=np.float32)
    else:
        command_arm = np.linalg.norm(np.diff(command[:, :6], axis=0), axis=1)
        command_gripper = np.abs(np.diff(command[:, 6], axis=0))
    if measured is None:
        measured_arm = np.zeros(transitions, dtype=np.float32)
        measured_gripper = np.zeros(transitions, dtype=np.float32)
    else:
        measured_arm = np.linalg.norm(np.diff(measured[:, :6], axis=0), axis=1)
        measured_gripper = np.abs(np.diff(measured[:, 6], axis=0))
    command_moving = ((command_arm > config.arm_delta_threshold)
                      | (command_gripper > config.gripper_delta_threshold))
    measured_moving = ((measured_arm > config.arm_delta_threshold)
                       | (measured_gripper > config.gripper_delta_threshold))
    if action_source in ("observation", "observation_ee"):
        moving = measured_moving
    elif action_source in {"action", "command"}:
        moving = command_moving
    else:
        moving = command_moving | measured_moving
    zeros = np.zeros(transitions, dtype=np.float32)
    return MotionSignals(command_arm, command_gripper, measured_arm, measured_gripper,
                         zeros, zeros, moving, command is not None, measured is not None, 1)


def detect_cutoff(signals: MotionSignals, length: int, config: CleanupConfig) -> int:
    """First sustained-motion onset, minus preroll, capped by minimum retention.

    Returns 0 (keep everything) for all-static episodes, episodes shorter than
    the confirmation window, or motion that starts after the scan limit.
    """

    scan = signals.moving[: config.max_scan_frames]
    window = config.confirmation_window_frames
    if len(scan) < window:
        return 0
    counts = np.convolve(scan.astype(np.int32), np.ones(window, dtype=np.int32), mode="valid")
    candidates = np.flatnonzero(counts >= config.confirmation_min_moving_frames)
    if not len(candidates):
        return 0
    window_start = int(candidates[0])
    first_moving = int(np.flatnonzero(scan[window_start: window_start + window])[0])
    onset = window_start + first_moving + signals.motion_frame_offset
    cutoff = max(0, onset - config.preroll_frames)
    return min(cutoff, max(0, length - config.min_remaining_frames))


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(np.mean(values)), "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values))}


def episode_metrics(episode_index: int, length: int, cutoff: int, fps: float,
                    signals: MotionSignals) -> dict[str, Any]:
    prefix_stop = max(0, cutoff - signals.motion_frame_offset)
    after_stop = min(len(signals.moving), cutoff + max(1, round(fps)))
    prefix_moving = signals.moving[:prefix_stop]
    after_moving = signals.moving[cutoff:after_stop]
    metrics: dict[str, Any] = {
        "episode_index": episode_index,
        "original_frames": length,
        "removed_frames": cutoff,
        "removed_seconds": cutoff / fps,
        "cleaned_frames": length - cutoff,
        "cleaned_seconds": (length - cutoff) / fps,
        "command_signal_available": signals.command_available,
        "measured_signal_available": signals.measured_available,
        "prefix_moving_fraction": float(np.mean(prefix_moving)) if len(prefix_moving) else 0.0,
        "first_second_moving_fraction_after_cleanup": (
            float(np.mean(after_moving)) if len(after_moving) else 0.0
        ),
    }
    for name in ("command_arm_delta", "command_gripper_delta", "measured_arm_delta",
                 "measured_gripper_delta", "ee_translation_delta", "ee_rotation_delta"):
        for statistic, value in _safe_stats(getattr(signals, name)[:prefix_stop]).items():
            metrics[f"prefix_{name}_{statistic}"] = value
    return metrics


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)), "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)), "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)), "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def summary(source: str, output_name: str, fps: float, rows: list[dict[str, Any]],
            config: CleanupConfig) -> dict[str, Any]:
    removed = np.asarray([row["removed_frames"] for row in rows], dtype=np.int64)
    original = int(sum(row["original_frames"] for row in rows))
    total_removed = int(removed.sum())
    return {
        "source_dataset": source,
        "output_dataset": output_name,
        "method": "first sustained target-action motion",
        "config": asdict(config),
        "episodes": len(rows),
        "original_frames": original,
        "removed_frames": total_removed,
        "remaining_frames": original - total_removed,
        "removed_fraction": total_removed / original if original else 0.0,
        "episodes_clipped": int(np.count_nonzero(removed)),
        "removed_frames_distribution": _quantiles(removed.astype(np.float64)),
        "removed_seconds_distribution": _quantiles(removed / fps),
        "mean_prefix_moving_fraction": float(np.mean([r["prefix_moving_fraction"] for r in rows])),
        "mean_first_second_moving_fraction_after_cleanup": float(
            np.mean([r["first_second_moving_fraction_after_cleanup"] for r in rows])
        ),
    }


def motion_source(setup: Setup, run: RunConfig) -> str:
    if run.cleanup_motion != "auto":
        return run.cleanup_motion
    return {"command": "command-joint", "delta-ee": "measured-ee",
            "delta-joint": "measured-delta-joint"}.get(setup.action, "measured-joint")


__all__ = ["CleanupConfig", "MotionSignals", "detect_cutoff", "episode_metrics",
           "motion_signals", "motion_source", "summary"]
