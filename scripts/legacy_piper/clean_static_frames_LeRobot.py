#!/usr/bin/env python3
"""Remove static episode prefixes from LeRobot v2.1 datasets.

The detector follows the dataset's action target: command motion for command-
action datasets, measured robot motion for feedback-action datasets, and the
already-relative translation/rotation action for measured delta-EE datasets.
A joint transition is moving when either the six-joint L2 delta exceeds
``--arm-delta-threshold`` or the gripper delta exceeds
``--gripper-delta-threshold``. A delta-EE action is moving when translation,
rotation, or measured gripper movement exceeds its corresponding threshold.
Cropping begins at the first moving action/frame in the first window containing
sustained motion.

For every input dataset this script writes:

* a complete ``<dataset>_cleanup`` LeRobot dataset with rewritten Parquet
  timestamps/indices, clipped videos, and updated metadata;
* ``meta/cleanup_metrics.csv`` with per-episode prefix measurements;
* ``meta/cleanup_summary.json`` with aggregate clipping metrics; and
* one labeled, side-by-side original/cleaned MP4 in ``Example``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _motion_signals(
    state: np.ndarray,
    action: np.ndarray,
    config: CleanupConfig,
    state_source: str | None = None,
    action_source: str | None = None,
    action_timestep_offset: int = 0,
) -> MotionSignals:
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
            raise ValueError("observation_delta_joint cleanup requires current measured joint/gripper state")
        measured_arm = np.linalg.norm(action[:, :6], axis=1)
        measured_gripper = np.abs(action[:, 6] - state[:, 6])
        moving = ((measured_arm > config.arm_delta_threshold)
                  | (measured_gripper > config.gripper_delta_threshold))
        zeros = np.zeros(len(action), dtype=np.float32)
        return MotionSignals(
            command_arm_delta=zeros, command_gripper_delta=zeros,
            measured_arm_delta=measured_arm, measured_gripper_delta=measured_gripper,
            ee_translation_delta=zeros, ee_rotation_delta=zeros,
            moving=moving, command_available=False, measured_available=True,
            motion_frame_offset=0,
        )

    if action_source == "observation_delta_ee":
        gripper_index = 6 if state_source in ("state", "state_last_command") else 7
        if state_source == "last_command" or state.shape[1] <= gripper_index:
            raise ValueError(
                "observation_delta_ee cleanup requires a current measured gripper "
                f"in observation.state, got state_source={state_source!r}, shape={state.shape}"
            )
        ee_translation = np.linalg.norm(action[:, :3], axis=1)
        ee_rotation = np.linalg.norm(action[:, 3:6], axis=1)
        measured_gripper = np.abs(action[:, 6] - state[:, gripper_index])
        moving = (
            (ee_translation > config.ee_translation_delta_threshold)
            | (ee_rotation > config.ee_rotation_delta_threshold)
            | (measured_gripper > config.gripper_delta_threshold)
        )
        zeros = np.zeros(len(action), dtype=np.float32)
        return MotionSignals(
            command_arm_delta=zeros,
            command_gripper_delta=zeros,
            measured_arm_delta=zeros,
            measured_gripper_delta=measured_gripper,
            ee_translation_delta=ee_translation,
            ee_rotation_delta=ee_rotation,
            moving=moving,
            command_available=False,
            measured_available=True,
            # Each row already stores motion beginning at that same frame:
            # measured EE[t] -> measured EE[t + 1].
            motion_frame_offset=0,
        )

    measured: np.ndarray | None = None
    command: np.ndarray | None = None
    if state_source == "state_last_command":
        if state.shape[1] != 14:
            raise ValueError(f"state_last_command requires 14 values, got {state.shape}")
        measured = state[:, :7]
        command = state[:, 7:14]
    elif state_source == "state":
        measured = state[:, :7]
    elif state_source == "last_command":
        command = state[:, :7]
    elif state_source is None:
        # Backward compatibility for datasets written before state_source was
        # recorded: the first seven state values are measured Piper feedback.
        measured = state[:, :7]
        if state.shape[1] >= 14:
            command = state[:, 7:14]

    if action_source == "observation" and action_timestep_offset == 0:
        measured = action
    elif action_source == "observation_ee":
        if measured is None:
            raise ValueError(
                "observation_ee cleanup requires measured joint state in observation.state"
            )
    elif action_source in {"action", "command"}:
        command = action
    elif action_source is None and command is None:
        command = action

    command_available = command is not None
    measured_available = measured is not None
    transition_count = len(state) - 1
    if command is None:
        command_arm = np.zeros(transition_count, dtype=np.float32)
        command_gripper = np.zeros(transition_count, dtype=np.float32)
    else:
        command_arm = np.linalg.norm(np.diff(command[:, :6], axis=0), axis=1)
        command_gripper = np.abs(np.diff(command[:, 6], axis=0))
    if measured is None:
        measured_arm = np.zeros(transition_count, dtype=np.float32)
        measured_gripper = np.zeros(transition_count, dtype=np.float32)
    else:
        measured_arm = np.linalg.norm(np.diff(measured[:, :6], axis=0), axis=1)
        measured_gripper = np.abs(np.diff(measured[:, 6], axis=0))
    command_moving = (
        (command_arm > config.arm_delta_threshold)
        | (command_gripper > config.gripper_delta_threshold)
    )
    measured_moving = (
        (measured_arm > config.arm_delta_threshold)
        | (measured_gripper > config.gripper_delta_threshold)
    )
    if action_source == "observation":
        moving = measured_moving
    elif action_source == "observation_ee":
        # Absolute EE pose differences are vulnerable to quaternion sign flips.
        # The measured joint/gripper state is an equivalent, stable motion signal
        # and also keeps clips identical to the measured-joint-action dataset.
        moving = measured_moving
    elif action_source in {"action", "command"}:
        moving = command_moving
    else:
        moving = command_moving | measured_moving
    return MotionSignals(
        command_arm_delta=command_arm,
        command_gripper_delta=command_gripper,
        measured_arm_delta=measured_arm,
        measured_gripper_delta=measured_gripper,
        ee_translation_delta=np.zeros(transition_count, dtype=np.float32),
        ee_rotation_delta=np.zeros(transition_count, dtype=np.float32),
        moving=moving,
        command_available=command_available,
        measured_available=measured_available,
        motion_frame_offset=1,
    )


def _detect_cutoff(signals: MotionSignals, length: int, config: CleanupConfig) -> int:
    scan = signals.moving[: config.max_scan_frames]
    window = config.confirmation_window_frames
    if len(scan) < window:
        return 0
    moving_counts = np.convolve(
        scan.astype(np.int32), np.ones(window, dtype=np.int32), mode="valid"
    )
    candidates = np.flatnonzero(
        moving_counts >= config.confirmation_min_moving_frames
    )
    if not len(candidates):
        return 0
    window_start = int(candidates[0])
    first_moving = int(np.flatnonzero(scan[window_start : window_start + window])[0])
    onset_frame = window_start + first_moving + signals.motion_frame_offset
    cutoff = max(0, onset_frame - config.preroll_frames)
    return min(cutoff, max(0, length - config.min_remaining_frames))


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _episode_metrics(
    episode_index: int,
    length: int,
    cutoff: int,
    fps: float,
    signals: MotionSignals,
) -> dict[str, Any]:
    # Ordinary joint deltas lead into frame i + 1, while a delta-EE action is
    # stored on the current frame i. Count only motion wholly inside the
    # removed prefix for either convention.
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
    signal_values = {
        "command_arm_delta": signals.command_arm_delta,
        "command_gripper_delta": signals.command_gripper_delta,
        "measured_arm_delta": signals.measured_arm_delta,
        "measured_gripper_delta": signals.measured_gripper_delta,
        "ee_translation_delta": signals.ee_translation_delta,
        "ee_rotation_delta": signals.ee_rotation_delta,
    }
    for name, values in signal_values.items():
        for statistic, value in _safe_stats(values[:prefix_stop]).items():
            metrics[f"prefix_{name}_{statistic}"] = value
    return metrics


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Parquet table is missing required column {name!r}")
    return table.set_column(column_index, name, values)


def _clipped_table(
    table: pa.Table,
    cutoff: int,
    episode_index: int,
    global_offset: int,
    fps: float,
) -> pa.Table:
    clipped = table.slice(cutoff)
    length = len(clipped)
    clipped = _replace_column(
        clipped,
        "timestamp",
        pa.array(np.arange(length, dtype=np.float32) / fps, type=pa.float32()),
    )
    clipped = _replace_column(
        clipped,
        "frame_index",
        pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
    )
    clipped = _replace_column(
        clipped,
        "episode_index",
        pa.array(np.full(length, episode_index, dtype=np.int64), type=pa.int64()),
    )
    clipped = _replace_column(
        clipped,
        "index",
        pa.array(
            np.arange(global_offset, global_offset + length, dtype=np.int64),
            type=pa.int64(),
        ),
    )
    return clipped


def _video_properties(path: Path) -> tuple[int, int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video {path}")
    properties = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    return properties


def _ffmpeg_path(explicit: Path | None) -> str:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--ffmpeg does not exist: {path}")
        return str(path)
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured and Path(configured).is_file():
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        raise RuntimeError(
            "ffmpeg is required; install it, set IMAGEIO_FFMPEG_EXE, or pass --ffmpeg"
        ) from None


def _encode_frames(
    ffmpeg: str,
    destination: Path,
    width: int,
    height: int,
    fps: float,
    frames: Sequence[np.ndarray] | Any,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stderr is not None
    count = 0
    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"frame {count} has shape {frame.shape}, expected {(height, width, 3)}"
                )
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            count += 1
        process.stdin.close()
        error_text = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {destination}: {error_text.strip()}")
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        destination.unlink(missing_ok=True)
        raise
    return count


def _clip_video(
    source: Path,
    destination: Path,
    cutoff: int,
    expected_original_frames: int,
    expected_cleaned_frames: int,
    fps: float,
    ffmpeg: str,
) -> None:
    width, height, frames, source_fps = _video_properties(source)
    if frames != expected_original_frames or not np.isclose(source_fps, fps, atol=1e-3):
        raise ValueError(
            f"{source}: expected {expected_original_frames} frames at {fps:g} Hz, "
            f"got {frames} at {source_fps:g} Hz"
        )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open video {source}")

    def frames_after_cutoff():
        source_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index >= cutoff:
                yield frame
            source_index += 1

    try:
        encoded = _encode_frames(
            ffmpeg, destination, width, height, fps, frames_after_cutoff()
        )
    finally:
        capture.release()
    if encoded != expected_cleaned_frames:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"{source}: encoded {encoded} frames, expected {expected_cleaned_frames}"
        )
    _, _, checked_frames, checked_fps = _video_properties(destination)
    if checked_frames != expected_cleaned_frames or not np.isclose(
        checked_fps, fps, atol=1e-3
    ):
        raise RuntimeError(
            f"{destination}: post-encode check got {checked_frames} frames at "
            f"{checked_fps:g} Hz"
        )


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _summary(
    input_path: Path,
    output_name: str,
    fps: float,
    rows: list[dict[str, Any]],
    config: CleanupConfig,
) -> dict[str, Any]:
    removed_frames = np.asarray([row["removed_frames"] for row in rows], dtype=np.int64)
    removed_seconds = removed_frames / fps
    original_frames = int(sum(row["original_frames"] for row in rows))
    total_removed = int(removed_frames.sum())
    return {
        "source_dataset": str(input_path),
        "output_dataset": output_name,
        "method": "first sustained target-action motion",
        "config": asdict(config),
        "episodes": len(rows),
        "original_frames": original_frames,
        "removed_frames": total_removed,
        "remaining_frames": original_frames - total_removed,
        "removed_fraction": total_removed / original_frames,
        "episodes_clipped": int(np.count_nonzero(removed_frames)),
        "removed_frames_distribution": _quantiles(removed_frames.astype(np.float64)),
        "removed_seconds_distribution": _quantiles(removed_seconds),
        "mean_prefix_moving_fraction": float(
            np.mean([row["prefix_moving_fraction"] for row in rows])
        ),
        "mean_first_second_moving_fraction_after_cleanup": float(
            np.mean(
                [row["first_second_moving_fraction_after_cleanup"] for row in rows]
            )
        ),
    }


def _comparison_video(
    original: Path,
    cleaned: Path,
    destination: Path,
    cutoff: int,
    fps: float,
    seconds: float,
    ffmpeg: str,
) -> None:
    left = cv2.VideoCapture(str(original))
    right = cv2.VideoCapture(str(cleaned))
    if not left.isOpened() or not right.isOpened():
        left.release()
        right.release()
        raise ValueError(f"could not open comparison inputs {original} and {cleaned}")
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (
        int(right.get(cv2.CAP_PROP_FRAME_WIDTH)) != width
        or int(right.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height
    ):
        left.release()
        right.release()
        raise ValueError("original and cleaned example resolutions differ")
    limit = min(
        round(seconds * fps),
        int(left.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(right.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    banner = 64

    def comparison_frames():
        try:
            for index in range(limit):
                left_ok, left_frame = left.read()
                right_ok, right_frame = right.read()
                if not left_ok or not right_ok:
                    break
                canvas = np.zeros((height + banner, width * 2, 3), dtype=np.uint8)
                canvas[banner:, :width] = left_frame
                canvas[banner:, width:] = right_frame
                original_time = index / fps
                cleaned_source_time = (index + cutoff) / fps
                cv2.putText(
                    canvas,
                    f"ORIGINAL  t={original_time:4.1f}s  (idle prefix)",
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"CLEANUP  source t={cleaned_source_time:4.1f}s",
                    (width + 16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 255, 80),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    f"Removed first {cutoff} frames = {cutoff / fps:.2f}s",
                    (16, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 210, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(canvas, (width, 0), (width, height + banner), (255, 255, 255), 2)
                yield canvas
        finally:
            left.release()
            right.release()

    encoded = _encode_frames(
        ffmpeg,
        destination,
        width * 2,
        height + banner,
        fps,
        comparison_frames(),
    )
    if encoded != limit:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"comparison encoded {encoded} frames, expected {limit}")


def clean_dataset(
    input_path: Path,
    output_root: Path,
    examples_root: Path,
    output_name: str | None,
    config: CleanupConfig,
    example_seconds: float,
    ffmpeg: str,
    overwrite: bool,
) -> Path:
    input_path = input_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    examples_root = examples_root.expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"input dataset does not exist: {input_path}")
    info_path = input_path / "meta/info.json"
    episodes_path = input_path / "meta/episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise ValueError(f"{input_path} is missing LeRobot metadata")

    info = _read_json(info_path)
    fps = float(info["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{info_path}: invalid fps {fps}")
    name = output_name or f"{input_path.name}_cleanup"
    if not name or Path(name).name != name:
        raise ValueError("output dataset name must be one directory name")
    output_path = output_root / name
    example_path = examples_root / f"{name}_original_vs_cleanup.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}; pass --overwrite")
    if example_path.exists() and not overwrite:
        raise FileExistsError(f"example exists: {example_path}; pass --overwrite")

    parquet_paths = sorted((input_path / "data").rglob("*.parquet"))
    episode_records = _jsonl(episodes_path)
    if len(parquet_paths) != len(episode_records) or not parquet_paths:
        raise ValueError(
            f"Parquet/episode metadata count mismatch: {len(parquet_paths)} vs "
            f"{len(episode_records)}"
        )
    camera_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    if not camera_keys:
        raise ValueError(f"{info_path}: no video features")

    tables: list[pa.Table] = []
    metrics: list[dict[str, Any]] = []
    for episode_index, parquet_path in enumerate(parquet_paths):
        table = pq.read_table(parquet_path)
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{parquet_path}: state/action contains NaN or Inf")
        signals = _motion_signals(
            state,
            action,
            config,
            state_source=info.get("state_source"),
            action_source=info.get("action_source"),
            action_timestep_offset=int(info.get("action_timestep_offset", 0)),
        )
        cutoff = _detect_cutoff(signals, len(table), config)
        tables.append(table)
        metrics.append(
            _episode_metrics(episode_index, len(table), cutoff, fps, signals)
        )

    summary = _summary(input_path, name, fps, metrics, config)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=output_root))
    try:
        # Preserve task/embodiment/modality metadata, then replace every item
        # that depends on episode lengths or indices.
        shutil.copytree(input_path / "meta", staging / "meta", dirs_exist_ok=True)
        global_offset = 0
        rewritten_episodes: list[str] = []
        for episode_index, (source_parquet, table, row) in enumerate(
            zip(parquet_paths, tables, metrics)
        ):
            relative_parquet = source_parquet.relative_to(input_path)
            destination_parquet = staging / relative_parquet
            destination_parquet.parent.mkdir(parents=True, exist_ok=True)
            clipped = _clipped_table(
                table,
                int(row["removed_frames"]),
                episode_index,
                global_offset,
                fps,
            )
            pq.write_table(clipped, destination_parquet)
            global_offset += len(clipped)

            record = dict(episode_records[episode_index])
            record["episode_index"] = episode_index
            record["length"] = len(clipped)
            rewritten_episodes.append(json.dumps(record))

            for camera_key in camera_keys:
                relative_video = Path(
                    info["video_path"].format(
                        episode_chunk=episode_index // int(info["chunks_size"]),
                        video_key=camera_key,
                        episode_index=episode_index,
                    )
                )
                _clip_video(
                    input_path / relative_video,
                    staging / relative_video,
                    int(row["removed_frames"]),
                    int(row["original_frames"]),
                    int(row["cleaned_frames"]),
                    fps,
                    ffmpeg,
                )
            print(
                f"[{episode_index + 1}/{len(tables)}] {input_path.name} episode "
                f"{episode_index}: removed {row['removed_frames']} frames "
                f"({row['removed_seconds']:.2f}s)"
            )

        updated_info = dict(info)
        updated_info["total_frames"] = int(summary["remaining_frames"])
        updated_info["cleanup"] = {
            "source_dataset": input_path.name,
            "method": summary["method"],
            "removed_frames": summary["removed_frames"],
            "removed_fraction": summary["removed_fraction"],
            "config": summary["config"],
        }
        _write_json(staging / "meta/info.json", updated_info)
        (staging / "meta/episodes.jsonl").write_text(
            "\n".join(rewritten_episodes) + "\n", encoding="utf-8"
        )
        _write_metrics_csv(staging / "meta/cleanup_metrics.csv", metrics)
        _write_json(staging / "meta/cleanup_summary.json", summary)

        if output_path.exists():
            shutil.rmtree(output_path)
        staging.replace(output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    example_episode = int(
        max(metrics, key=lambda row: int(row["removed_frames"]))["episode_index"]
    )
    example_cutoff = int(metrics[example_episode]["removed_frames"])
    example_chunk = example_episode // int(info["chunks_size"])
    example_camera = (
        "observation.images.global"
        if "observation.images.global" in camera_keys
        else camera_keys[0]
    )
    relative_example = Path(
        info["video_path"].format(
            episode_chunk=example_chunk,
            video_key=example_camera,
            episode_index=example_episode,
        )
    )
    examples_root.mkdir(parents=True, exist_ok=True)
    temporary_example = examples_root / f".{example_path.name}.tmp.mp4"
    temporary_example.unlink(missing_ok=True)
    try:
        _comparison_video(
            input_path / relative_example,
            output_path / relative_example,
            temporary_example,
            example_cutoff,
            fps,
            example_seconds,
            ffmpeg,
        )
        if example_path.exists():
            example_path.unlink()
        temporary_example.replace(example_path)
    except Exception:
        temporary_example.unlink(missing_ok=True)
        raise

    print(
        f"Completed {name}: removed {summary['removed_frames']} / "
        f"{summary['original_frames']} frames ({summary['removed_fraction']:.2%}); "
        f"example episode={example_episode}, cutoff={example_cutoff / fps:.2f}s"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        type=Path,
        help="One or more converted LeRobot dataset directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Parent directory for DATASETNAME_cleanup outputs.",
    )
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=None,
        help="Comparison-video directory (default: OUTPUT_ROOT/Example).",
    )
    parser.add_argument(
        "--output-name",
        help="Custom output name; valid only when processing one dataset.",
    )
    parser.add_argument("--arm-delta-threshold", type=float, default=0.005)
    parser.add_argument("--gripper-delta-threshold", type=float, default=0.01)
    parser.add_argument("--ee-translation-delta-threshold", type=float, default=0.0005)
    parser.add_argument("--ee-rotation-delta-threshold", type=float, default=0.005)
    parser.add_argument("--confirmation-window-seconds", type=float, default=0.4)
    parser.add_argument("--confirmation-min-fraction", type=float, default=0.5)
    parser.add_argument("--preroll-seconds", type=float, default=0.0)
    parser.add_argument("--max-scan-seconds", type=float, default=10.0)
    parser.add_argument("--min-remaining-seconds", type=float, default=2.0)
    parser.add_argument("--example-seconds", type=float, default=6.0)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.output_name and len(args.datasets) != 1:
        raise ValueError("--output-name can only be used with one input dataset")
    if (
        args.arm_delta_threshold <= 0
        or args.gripper_delta_threshold <= 0
        or args.ee_translation_delta_threshold <= 0
        or args.ee_rotation_delta_threshold <= 0
    ):
        raise ValueError("motion thresholds must be positive")
    if not 0 < args.confirmation_min_fraction <= 1:
        raise ValueError("--confirmation-min-fraction must be in (0, 1]")
    for name in (
        "confirmation_window_seconds",
        "max_scan_seconds",
        "min_remaining_seconds",
        "example_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.preroll_seconds < 0:
        raise ValueError("--preroll-seconds must be non-negative")

    ffmpeg = _ffmpeg_path(args.ffmpeg)
    output_root = args.output_root.expanduser().resolve()
    examples_root = (
        args.examples_root.expanduser().resolve()
        if args.examples_root is not None
        else output_root / "Example"
    )
    for dataset in args.datasets:
        info = _read_json(dataset.expanduser().resolve() / "meta/info.json")
        fps = float(info["fps"])
        window = max(1, round(args.confirmation_window_seconds * fps))
        config = CleanupConfig(
            arm_delta_threshold=args.arm_delta_threshold,
            gripper_delta_threshold=args.gripper_delta_threshold,
            ee_translation_delta_threshold=args.ee_translation_delta_threshold,
            ee_rotation_delta_threshold=args.ee_rotation_delta_threshold,
            confirmation_window_frames=window,
            confirmation_min_moving_frames=max(
                1, round(args.confirmation_min_fraction * window)
            ),
            preroll_frames=round(args.preroll_seconds * fps),
            max_scan_frames=round(args.max_scan_seconds * fps),
            min_remaining_frames=round(args.min_remaining_seconds * fps),
        )
        clean_dataset(
            input_path=dataset,
            output_root=output_root,
            examples_root=examples_root,
            output_name=args.output_name,
            config=config,
            example_seconds=args.example_seconds,
            ffmpeg=ffmpeg,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
