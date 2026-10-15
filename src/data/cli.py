"""Piper raw recordings -> LeRobot v2.1 dataset -> optional B-spline sidecar.

Examples:
  spline-preprocess-piper --list-setups
  spline-preprocess-piper --config examples/piper_preprocess.json --dry-run
  spline-preprocess-piper --raw-root DATA/classify_blocks --setup state-delta-joint \\
      --output-root outputs/piper --spline --spline-mode uniform_left
  spline-preprocess-piper --raw-root DATA --tasks classify_blocks \\
      --state joint-last-command --action abs-ee --cameras global

Flags use the names and defaults of scripts/legacy_piper/preprocess_piper.py; flags given
on the command line override values from --config.  Images/state are at t;
absolute actions at t+offset (presets keep their historical offset); deltas
are t->t+1 with absolute measured gripper[t+1].
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import sys

from .config import (
    ACTIONS, CLEANUP_MOTIONS, PRESETS, STATES, ExtraObservation, PipelineConfig, Setup,
    default_camera, load_config,
)
from .pipeline import run


def _pairs(values: list[str] | None, label: str) -> dict[str, str]:
    result = {}
    for value in values or ():
        if "=" not in value:
            raise SystemExit(f"{label} expects NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        result[name.strip()] = item.strip()
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="JSON (or TOML on Python >= 3.11) pipeline config")
    p.add_argument("--list-setups", action="store_true")
    p.add_argument("--setup", choices=tuple(PRESETS), action="append",
                   help="Preset; repeat to run several")
    p.add_argument("--all-setups", action="store_true")
    p.add_argument("--state", choices=STATES)
    p.add_argument("--action", choices=ACTIONS)
    p.add_argument("--action-offset", type=int, choices=(0, 1))
    p.add_argument("--quaternion", choices=("raw", "continuous"))
    p.add_argument("--last-command-clock", choices=("raw", "target"))
    p.add_argument("--delta-ee-frame", choices=("tool", "base"))
    p.add_argument("--angle-unit", choices=("radian", "degree"))
    p.add_argument("--position-unit", choices=("meter", "millimeter"))
    p.add_argument("--raw-root", help="Parent of task folders, or one task folder")
    p.add_argument("--output-root")
    p.add_argument("--output", help="Explicit dataset directory (single job only)")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--task-description")
    p.add_argument("--cameras", help="Comma-separated camera names; empty string for none")
    p.add_argument("--camera-key", action="append", metavar="NAME=KEY",
                   help="LeRobot key for a selected camera")
    p.add_argument("--extra-observation", action="append", metavar="KEY=FIELD",
                   help="e.g. observation.joint_velocity=arm_joint_velocity")
    p.add_argument("--target-fps", type=float)
    p.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cleanup-motion", choices=CLEANUP_MOTIONS)
    p.add_argument("--arm-delta-threshold", type=float)
    p.add_argument("--gripper-delta-threshold", type=float)
    p.add_argument("--ee-translation-delta-threshold", type=float)
    p.add_argument("--ee-rotation-delta-threshold", type=float)
    p.add_argument("--confirmation-window-seconds", type=float)
    p.add_argument("--confirmation-min-fraction", type=float)
    p.add_argument("--preroll-seconds", type=float)
    p.add_argument("--max-scan-seconds", type=float)
    p.add_argument("--min-remaining-seconds", type=float)
    p.add_argument("--validate", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--skip-video-validation", action="store_true")
    p.add_argument("--limit-episodes", type=int)
    p.add_argument("--jobs", type=int, help="Parallel episode writers")
    p.add_argument("--ffmpeg")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Align and crop, print the plan, write nothing")
    spline = p.add_argument_group("optional B-spline sidecar")
    spline.add_argument("--spline", action=argparse.BooleanOptionalAction, default=None)
    spline.add_argument("--spline-mode")
    spline.add_argument("--spline-preset")
    spline.add_argument("--spline-encoder-state")
    spline.add_argument("--spline-chunk-size", type=int)
    spline.add_argument("--spline-degree", type=int)
    spline.add_argument("--spline-num-basis", type=int)
    spline.add_argument("--spline-span-length-steps", type=int)
    spline.add_argument("--spline-vocab-size", type=int)
    spline.add_argument("--spline-regularization", type=float)
    spline.add_argument("--spline-fit-tolerance", type=float)
    spline.add_argument("--spline-exclude-knot-dimensions",
                        help="Comma-separated adaptive-only dimensions")
    spline.add_argument("--spline-quantize", action=argparse.BooleanOptionalAction, default=None)
    spline.add_argument("--spline-normalization", choices=("none", "q01_q99"))
    spline.add_argument("--spline-validation-episodes", type=int)
    spline.add_argument("--spline-output")
    return p


def config_from_args(args) -> PipelineConfig:
    config = load_config(args.config) if args.config else PipelineConfig()
    changes: dict = {}
    for name in ("raw_root", "output_root", "output", "task_description", "ffmpeg",
                 "limit_episodes", "jobs"):
        value = getattr(args, name)
        if value is not None:
            changes[name] = value
    if args.tasks:
        changes["tasks"] = tuple(args.tasks)
    if args.validate is not None:
        changes["validate"] = args.validate
    if args.skip_video_validation:
        changes["check_videos"] = False
    if args.overwrite:
        changes["overwrite"] = True
    if args.dry_run:
        changes["dry_run"] = True
    if args.all_setups and args.setup:
        raise SystemExit("Choose --all-setups or --setup, not both")
    overrides = {name: getattr(args, name) for name in (
        "state", "action", "action_offset", "quaternion", "last_command_clock",
        "delta_ee_frame", "angle_unit", "position_unit") if getattr(args, name) is not None}
    if args.all_setups or args.setup:
        changes["setups"] = tuple(PRESETS) if args.all_setups else tuple(args.setup)
        changes["custom"] = None
        changes["setup_overrides"] = {**config.setup_overrides, **overrides}
    elif overrides:
        if config.custom is not None:
            changes["custom"] = replace(config.custom, **overrides)
        elif config.setups:
            changes["setup_overrides"] = {**config.setup_overrides, **overrides}
        else:
            if "state" not in overrides or "action" not in overrides:
                raise SystemExit("Select --setup, --all-setups, or both --state and --action")
            changes["custom"] = Setup(**overrides)
    run_changes = {key: value for key, value in {
        "fps": args.target_fps, "cleanup": args.cleanup, "cleanup_motion": args.cleanup_motion,
        "arm_threshold": args.arm_delta_threshold,
        "gripper_threshold": args.gripper_delta_threshold,
        "translation_threshold": args.ee_translation_delta_threshold,
        "rotation_threshold": args.ee_rotation_delta_threshold,
        "confirmation_seconds": args.confirmation_window_seconds,
        "confirmation_fraction": args.confirmation_min_fraction,
        "preroll_seconds": args.preroll_seconds, "max_scan_seconds": args.max_scan_seconds,
        "min_remaining_seconds": args.min_remaining_seconds,
    }.items() if value is not None}
    if run_changes:
        changes["run"] = replace(config.run, **run_changes)
    cameras = config.cameras
    if args.cameras is not None:
        names = [name.strip() for name in args.cameras.split(",") if name.strip()]
        cameras = tuple(default_camera(name) for name in names)
    keys = _pairs(args.camera_key, "--camera-key")
    if keys:
        unknown = set(keys) - {camera.name for camera in cameras}
        if unknown:
            raise SystemExit(f"--camera-key names unselected cameras: {sorted(unknown)}")
        cameras = tuple(replace(camera, key=keys.get(camera.name, camera.key)) for camera in cameras)
    changes["cameras"] = cameras
    extras = _pairs(args.extra_observation, "--extra-observation")
    if extras:
        changes["extra_observations"] = tuple(ExtraObservation(k, v) for k, v in extras.items())
    spline_changes = {key: value for key, value in {
        "enabled": args.spline, "mode": args.spline_mode, "preset": args.spline_preset,
        "encoder_state": args.spline_encoder_state, "chunk_size": args.spline_chunk_size,
        "degree": args.spline_degree, "num_basis": args.spline_num_basis,
        "span_length_steps": args.spline_span_length_steps,
        "vocab_size": args.spline_vocab_size, "regularization": args.spline_regularization,
        "fit_tolerance": args.spline_fit_tolerance, "quantize": args.spline_quantize,
        "normalization": args.spline_normalization,
        "validation_episodes": args.spline_validation_episodes, "output": args.spline_output,
    }.items() if value is not None}
    if args.spline_exclude_knot_dimensions is not None:
        spline_changes["exclude_knot_dimensions"] = tuple(
            int(item) for item in args.spline_exclude_knot_dimensions.split(",") if item.strip())
    if spline_changes:
        if any(key != "enabled" for key in spline_changes) and args.spline is None:
            spline_changes["enabled"] = True
        changes["spline"] = replace(config.spline, **spline_changes)
    return replace(config, **changes)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_setups:
        for name, (setup, folder) in PRESETS.items():
            print(f"{name:30} state={setup.state:19} action={setup.action:11} "
                  f"offset={setup.action_offset} -> {folder}")
        return 0
    config = config_from_args(args)
    if not config.setups and config.custom is None:
        raise SystemExit("Select --setup, --all-setups, --state/--action, or a config file")
    print(json.dumps({"config": json.loads(json.dumps(asdict(config), default=str))},
                     indent=2), flush=True)
    reports = run(config, log=lambda text: print(text, flush=True))
    if not config.dry_run:
        print(json.dumps({"reports": reports}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
