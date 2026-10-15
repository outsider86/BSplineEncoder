"""One-command Piper raw recordings -> LeRobot v2.1 -> optional spline sidecar.

Every output is first written to a staging directory next to its
destination, validated, and only then swapped into place together with its
sidecar.  If anything fails, previously completed outputs are left untouched
and staging directories are removed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from .config import (
    ACTION_SOURCE, PRESETS, STATE_SOURCE, TASKS, PipelineConfig, Setup,
)
from .conversion.episode import Episode, clip_episode, load_episode
from .conversion.export import (
    ALIGNMENT_PATH, CHUNKS_SIZE, DATA_PATH, VIDEO_PATH, episode_path, features, frame_columns,
    numeric_stats, write_alignment_parquet, write_episode_parquet, write_json, write_jsonl,
    write_metrics_csv,
)
from .conversion.raw import discover_episodes
from .transforms.cleanup import motion_source, summary as cleanup_summary
from .transforms.spaces import apply_units, feature_names, modality

PREPROCESSING_VERSION = 2


@dataclass(frozen=True)
class Job:
    preset: str
    setup: Setup
    raw_task: Path
    output: Path
    task: str
    sidecar: Path | None


def _family(name: str, setup: Setup, config: PipelineConfig) -> str:
    if name in PRESETS:
        return PRESETS[name][1]
    return (f"Custom/{setup.state}__{setup.action}__offset{setup.action_offset}"
            f"__quat-{setup.quaternion}__last-{setup.last_command_clock}"
            f"__motion-{motion_source(setup, config.run)}")


def _overlaps(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def plan(config: PipelineConfig) -> list[Job]:
    """Resolve jobs and reject ambiguous, overlapping, or unsafe paths."""

    config.check()
    raw_root = Path(config.raw_root).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw root does not exist: {raw_root}")
    input_root = raw_root  # outputs must stay outside the tree the user supplied
    if not config.tasks and any(raw_root.glob("*/data.pkl")):
        # ``raw_root`` is itself one task folder of episode directories.
        tasks = (raw_root.name,)
        raw_root = raw_root.parent
    else:
        tasks = config.tasks or tuple(sorted(
            p.name for p in raw_root.iterdir() if p.is_dir() and discover_episodes(p)))
    if not tasks:
        raise ValueError(f"No tasks containing raw episodes under {raw_root}")
    setups: list[tuple[str, Setup]] = []
    for name in config.setups:
        base = PRESETS[name][0]
        setups.append((name, replace(base, **_custom_overrides(config, base))))
    if config.custom is not None:
        setups.append(("custom", config.custom))
    jobs = []
    for name, setup in setups:
        setup.check()
        for task_name in tasks:
            if Path(task_name).name != task_name or task_name in (".", ".."):
                raise ValueError("Each task must be one raw subfolder name")
            raw_task = raw_root / task_name
            if not raw_task.is_dir() or not discover_episodes(raw_task):
                raise ValueError(f"No raw episodes in {raw_task}")
            instruction = config.task_description or TASKS.get(task_name)
            if not instruction or not instruction.strip():
                raise ValueError(f"Unknown task {task_name}: provide a task description")
            if config.output is not None:
                output = Path(config.output).expanduser().resolve()
            else:
                dataset = f"{task_name}_{config.run.fps:g}hz" + ("_cleanup" if config.run.cleanup else "")
                output = Path(config.output_root).expanduser().resolve() / _family(name, setup, config) / dataset
            sidecar = None
            if config.spline.enabled:
                sidecar = (Path(config.spline.output).expanduser().resolve() if config.spline.output
                           else output.parent / f"{output.name}_spline_{config.spline.mode}")
            jobs.append(Job(name, setup, raw_task, output, instruction.strip(), sidecar))
    destinations = [job.output for job in jobs] + [job.sidecar for job in jobs if job.sidecar]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Duplicate output paths in this run")
    for index, first in enumerate(destinations):
        if _overlaps(first, input_root) or any(_overlaps(first, job.raw_task) for job in jobs):
            raise ValueError(f"Output {first} overlaps the raw input tree")
        for second in destinations[index + 1:]:
            if _overlaps(first, second):
                raise ValueError(f"Output paths overlap: {first} and {second}")
    if config.spline.enabled:
        # Fail before any work: action_dim/fps follow from the setup and clock.
        from .splines.sidecar import build_encoder

        for job in jobs:
            build_encoder(config.spline, len(feature_names(job.setup, "action")), config.run.fps)
    if not config.overwrite and not config.dry_run:
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Output exists: {path}; use --overwrite to replace")
    return jobs


def _custom_overrides(config: PipelineConfig, base: Setup) -> dict:
    # Legacy ``--setup NAME --action-offset 0`` style overrides keep the
    # preset's output family, exactly as in scripts/legacy_piper/preprocess_piper.py.
    allowed = {name for name in asdict(base)}
    unknown = set(config.setup_overrides) - allowed
    if unknown:
        raise ValueError(f"unknown setup overrides: {sorted(unknown)}")
    return dict(config.setup_overrides)


def _sources(job: Job, config: PipelineConfig) -> list[Path]:
    sources = discover_episodes(job.raw_task)
    if config.limit_episodes is not None:
        sources = sources[: config.limit_episodes]
    if not sources:
        raise ValueError(f"No raw episodes under {job.raw_task}")
    return sources


def align_job(job: Job, config: PipelineConfig) -> tuple[list[Episode], list[dict], list[dict]]:
    """Load, resample, construct spaces, and crop every episode (no writing)."""

    episodes, metrics, selection = [], [], []
    for index, path in enumerate(_sources(job, config)):
        aligned = load_episode(path, job.setup, config.run, config.cameras, config.extra_observations)
        episode, metric = clip_episode(aligned, job.setup, config.run, index)
        if episode.length < 1:
            raise ValueError(f"{path}: no rows remain after alignment and cleanup")
        if episodes:
            reference = episodes[0]
            for name, stream in episode.cameras.items():
                expected = reference.cameras[name]
                if (stream.width, stream.height) != (expected.width, expected.height):
                    raise ValueError(f"{path}: {name} resolution changed")
            for key, values in episode.extras.items():
                if values.shape[1] != reference.extras[key].shape[1]:
                    raise ValueError(f"{path}: {key} dimension changed")
        episodes.append(episode)
        metrics.append(metric)
        selection.append({"episode_index": index, "source_episode": str(path.resolve()),
                          "resampled_frames": len(aligned.robot_indices),
                          "dropped_tail_frames": job.setup.action_offset,
                          "removed_prefix_frames": metric["removed_frames"]})
    return episodes, metrics, selection


def _write_episode(staging: Path, episode: Episode, index: int, offset: int, setup: Setup,
                   config: PipelineConfig, ffmpeg: str | None) -> dict:
    from .conversion.media import write_resampled_video

    state = apply_units(episode.state, setup, "state")
    action = apply_units(episode.action, setup, "action")
    columns = frame_columns(episode, index, offset, state, action)
    write_episode_parquet(staging / episode_path(DATA_PATH, index), columns)
    write_alignment_parquet(staging / episode_path(ALIGNMENT_PATH, index), episode)
    stats = {key: numeric_stats(values) for key, values in columns.items()}
    for camera in config.cameras:
        stats[camera.key] = write_resampled_video(
            episode.cameras[camera.name], episode.camera_indices[camera.name],
            staging / episode_path(VIDEO_PATH, index, camera.key), config.run.fps, ffmpeg)
    return {"episode_index": index, "stats": stats}


def _metadata(staging: Path, job: Job, config: PipelineConfig, episodes: list[Episode],
              metrics: list[dict], selection: list[dict], stats: list[dict]) -> dict:
    setup, run = job.setup, config.run
    total = sum(episode.length for episode in episodes)
    count = len(episodes)
    reference = episodes[0]
    state_mod, action_mod, state_layout, action_layout = modality(setup)
    camera_keys = {camera.name: camera.key for camera in config.cameras}
    info = {
        "codebase_version": "v2.1", "robot_type": "piper", "fps": run.fps,
        "total_episodes": count, "total_frames": total, "total_tasks": 1,
        "total_videos": count * len(config.cameras),
        "total_chunks": (count - 1) // CHUNKS_SIZE + 1, "chunks_size": CHUNKS_SIZE,
        "splits": _splits(count, config.spline.validation_episodes),
        "data_path": DATA_PATH, "video_path": VIDEO_PATH if config.cameras else None,
        "features": features(
            {name: (stream.height, stream.width) for name, stream in reference.cameras.items()},
            camera_keys, run.fps, feature_names(setup, "state"), feature_names(setup, "action"),
            {key: values.shape[1] for key, values in reference.extras.items()}),
        "state_source": STATE_SOURCE[setup.state], "action_source": ACTION_SOURCE[setup.action],
        "state_layout": state_layout, "action_layout": action_layout,
        "action_timestep_offset": setup.action_offset,
        "preprocessing": {
            "version": PREPROCESSING_VERSION, "package": "data",
            "preset": job.preset, "setup": asdict(setup), "run": asdict(run),
            "cameras": [asdict(camera) for camera in config.cameras],
            "extra_observations": [asdict(extra) for extra in config.extra_observations],
            "cleanup_motion_source": motion_source(setup, run),
            "raw_root": str(job.raw_task.resolve()), "limit_episodes": config.limit_episodes,
        },
    }
    if len(info["features"]["observation.state"]["names"]) != reference.state.shape[1] or \
            len(info["features"]["action"]["names"]) != reference.action.shape[1]:
        raise RuntimeError("feature names do not match exported dimensions")
    detector = run.detector()
    summary = cleanup_summary(str(job.raw_task.resolve()), job.output.name, run.fps, metrics, detector)
    summary["method"] = "first sustained motion" if run.cleanup else "disabled"
    summary["motion_source"] = motion_source(setup, run)
    info["cleanup"] = {key: summary[key] for key in
                       ("method", "removed_frames", "config", "motion_source")}
    info["cleanup"]["enabled"] = run.cleanup
    write_json(staging / "meta/info.json", info)
    write_json(staging / "meta/modality.json", {
        "video": {camera.name: {"original_key": camera.key} for camera in config.cameras},
        "state": state_mod, "action": action_mod,
        "annotation": {"human.action.task_description": {"original_key": "task_index"}},
    })
    write_json(staging / "meta/embodiment.json", {
        "robot_name": "Piper", "robot_type": "piper", "embodiment_tag": "new_embodiment",
        "record_frequency": run.fps, "body_controller_frequency": run.fps,
        "hand_controller_frequency": run.fps, "state_source": info["state_source"],
        "action_source": info["action_source"], "state_layout": state_layout,
        "action_layout": action_layout, "action_timestep_offset": setup.action_offset,
    })
    write_json(staging / "meta/cleanup_summary.json", summary)
    write_metrics_csv(staging / "meta/cleanup_metrics.csv", metrics)
    write_jsonl(staging / "meta/episodes.jsonl",
                [{"episode_index": i, "tasks": [job.task], "length": e.length}
                 for i, e in enumerate(episodes)])
    write_jsonl(staging / "meta/tasks.jsonl", [{"task_index": 0, "task": job.task}])
    write_jsonl(staging / "meta/episodes_stats.jsonl", stats)
    write_jsonl(staging / "meta/source_episodes.jsonl", selection)
    return info


def _splits(count: int, validation: int) -> dict[str, str]:
    if not validation:
        return {"train": f"0:{count}"}
    if validation >= count:
        raise ValueError("validation_episodes must leave at least one training episode")
    return {"train": f"0:{count - validation}", "validation": f"{count - validation}:{count}"}


def _swap(pairs: list[tuple[Path, Path]]) -> None:
    """Move staged outputs into place; restore every original on failure."""

    backups: list[tuple[Path, Path]] = []
    placed: list[Path] = []
    try:
        for staged, destination in pairs:
            if destination.exists():
                backup = Path(tempfile.mkdtemp(prefix=f".{destination.name}_backup_",
                                               dir=destination.parent))
                backup.rmdir()
                destination.replace(backup)
                backups.append((backup, destination))
            staged.replace(destination)
            placed.append(destination)
    except BaseException:
        for destination in placed:
            shutil.rmtree(destination, ignore_errors=True)
        for backup, destination in backups:
            backup.replace(destination)
        raise
    for backup, _ in backups:
        shutil.rmtree(backup, ignore_errors=True)


def convert(job: Job, config: PipelineConfig, ffmpeg: str | None = None,
            log=print) -> dict[str, Any]:
    """Run one job end to end; return a JSON-serializable report."""

    from .validation.dataset import validate_dataset

    episodes, metrics, selection = align_job(job, config)
    _splits(len(episodes), config.spline.validation_episodes)
    if config.cameras and ffmpeg is None:
        from .conversion.media import resolve_ffmpeg

        ffmpeg = resolve_ffmpeg(config.ffmpeg)
    staged: list[tuple[Path, Path]] = []
    try:
        job.output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{job.output.name}_", dir=job.output.parent))
        staged.append((staging, job.output))
        offsets = np.concatenate([[0], np.cumsum([e.length for e in episodes])[:-1]]).astype(int)
        arguments = [(staging, episode, index, int(offsets[index]), job.setup, config, ffmpeg)
                     for index, episode in enumerate(episodes)]
        if config.jobs == 1:
            stats = [_write_episode(*item) for item in arguments]
        else:
            with ThreadPoolExecutor(max_workers=config.jobs) as pool:
                stats = list(pool.map(lambda item: _write_episode(*item), arguments))
        for index, episode in enumerate(episodes):
            log(f"[{job.preset} {job.raw_task.name} {index + 1}/{len(episodes)}] "
                f"{episode.source_dir.name}: {episode.length} frames, removed "
                f"{metrics[index]['removed_frames']}")
        _metadata(staging, job, config, episodes, metrics, selection, stats)
        report: dict[str, Any] = {"dataset": str(job.output), "episodes": len(episodes),
                                  "frames": int(sum(e.length for e in episodes))}
        if config.validate:
            report["validation"] = validate_dataset(staging, config.check_videos)
            write_json(staging / "meta/validation.json", report["validation"])
        if job.sidecar is not None:
            from .splines.sidecar import encode_sidecar

            job.sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar_staging = Path(tempfile.mkdtemp(prefix=f".{job.sidecar.name}_",
                                                    dir=job.sidecar.parent))
            staged.append((sidecar_staging, job.sidecar))
            sidecar_staging.rmdir()  # encode_lerobot_dataset creates it
            quaternion = (3, 7) if job.setup.action == "abs-ee" else None
            report["sidecar"] = encode_sidecar(
                staging, sidecar_staging, config.spline, final_dataset=job.output,
                quaternion_columns=quaternion,
                provenance={"raw_task": str(job.raw_task.resolve()), "preset": job.preset,
                            "setup": asdict(job.setup),
                            "source_episodes": "meta/source_episodes.jsonl in the dataset"})
            report["sidecar"]["path"] = str(job.sidecar)
            if "warning" in report["sidecar"]["verification"]:
                log(f"WARNING {job.sidecar.name}: {report['sidecar']['verification']['warning']}")
        _swap(staged)
    except BaseException:
        for path, _ in staged:
            shutil.rmtree(path, ignore_errors=True)
        raise
    log(f"Saved {job.output}: {report['episodes']} episodes / {report['frames']} frames"
        + (f"; sidecar {job.sidecar}" if job.sidecar else ""))
    return report


def dry_run(job: Job, config: PipelineConfig) -> dict[str, Any]:
    """Align and crop without writing anything; report planned lengths."""

    episodes, metrics, _ = align_job(job, config)
    return {"preset": job.preset, "raw": str(job.raw_task), "output": str(job.output),
            "sidecar": None if job.sidecar is None else str(job.sidecar),
            "setup": asdict(job.setup), "task": job.task, "episodes": len(episodes),
            "frames": int(sum(e.length for e in episodes)),
            "removed_frames": int(sum(m["removed_frames"] for m in metrics)),
            "state_dim": int(episodes[0].state.shape[1]),
            "action_dim": int(episodes[0].action.shape[1]),
            "cameras": [camera.name for camera in config.cameras]}


def run(config: PipelineConfig, log=print) -> list[dict[str, Any]]:
    jobs = plan(config)
    if config.dry_run:
        reports = [dry_run(job, config) for job in jobs]
        for report in reports:
            log(json.dumps(report, indent=2))
        return reports
    ffmpeg = None
    if config.cameras:
        from .conversion.media import resolve_ffmpeg

        ffmpeg = resolve_ffmpeg(config.ffmpeg)
        os.environ.setdefault("IMAGEIO_FFMPEG_EXE", ffmpeg)
    return [convert(job, config, ffmpeg, log) for job in jobs]


__all__ = ["Job", "align_job", "convert", "dry_run", "plan", "run"]
