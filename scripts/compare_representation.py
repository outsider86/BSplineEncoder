#!/usr/bin/env python3
"""Compare adaptive-left and uniform-left representation quality at equal budget."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

from spline_encoder import ActionAlignment, create_encoder, load_encoder, load_lerobot_episodes
from spline_encoder.adaptive import fit_adaptive_bspline


def _metrics(errors: list[np.ndarray], gripper_threshold: float) -> dict:
    values = np.concatenate(errors, axis=0)
    absolute = np.abs(values)
    return {
        "samples": int(values.shape[0]),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "arm_rmse": float(np.sqrt(np.mean(np.square(values[:, :6])))),
        "gripper_rmse": float(np.sqrt(np.mean(np.square(values[:, 6])))),
        "mae": float(absolute.mean()),
        "max_absolute_error": float(absolute.max()),
        "gripper_threshold": gripper_threshold,
    }


def _stored_errors(episodes, encoded: Path, horizon: int):
    errors = []
    gripper_matches = 0
    gripper_count = 0
    padded = 0
    for episode in episodes:
        path = encoded / "data" / f"episode_{episode.episode_index:06d}.npz"
        with np.load(path) as stored:
            controls = np.asarray(stored["controls"], dtype=np.float64)
            knots = np.asarray(stored["knots"], dtype=np.float64)
            executable = np.asarray(stored["executable_steps"], dtype=np.int64)
        for observation, (control, knot, native_horizon) in enumerate(
            zip(controls, knots, executable)
        ):
            steps = np.minimum(np.arange(horizon), native_horizon - 1)
            decoded = BSpline(knot, control, 3, extrapolate=False)(steps / 25.0)
            target_indices = np.minimum(
                observation + np.arange(horizon), len(episode.actions) - 1
            )
            target = episode.actions[target_indices]
            errors.append(decoded - target)
            gripper_matches += int(np.count_nonzero((decoded[:, 6] >= 0.3) == (target[:, 6] >= 0.3)))
            gripper_count += horizon
            padded += int(native_horizon < horizon)
    return errors, gripper_matches / gripper_count, padded


def _global_equal_budget_errors(episodes, encoder):
    adaptive_errors = []
    uniform_errors = []
    adaptive_binary_matches = 0
    uniform_binary_matches = 0
    gripper_count = 0
    selected_controls = []
    maximum_controls = []
    for episode in episodes:
        timestamps = np.arange(len(episode.actions), dtype=np.float64) * encoder.config.delta_t
        cap = encoder.max_episode_control_points(len(episode.actions))
        fit = fit_adaptive_bspline(
            episode.actions,
            timestamps=timestamps,
            degree=encoder.config.degree,
            max_error=encoder.config.fit_tolerance,
            smoothing=encoder.config.smoothing,
            max_control_points=cap,
            knot_insertion_excluded_dimensions=encoder.config.knot_insertion_excluded_dimensions,
        )
        selected_controls.append(len(fit.control_points))
        maximum_controls.append(cap)
        adaptive_decoded = fit.spline(timestamps)
        spans = cap - encoder.config.degree
        breaks = np.linspace(timestamps[0], timestamps[-1], spans + 1)
        uniform_knots = np.concatenate((
            np.repeat(breaks[0], encoder.config.degree),
            breaks,
            np.repeat(breaks[-1], encoder.config.degree),
        ))
        uniform_spline = make_lsq_spline(
            timestamps,
            episode.actions,
            uniform_knots,
            k=encoder.config.degree,
        )
        uniform_decoded = uniform_spline(timestamps)
        adaptive_errors.append(adaptive_decoded - episode.actions)
        uniform_errors.append(uniform_decoded - episode.actions)
        adaptive_binary_matches += int(np.count_nonzero(
            (adaptive_decoded[:, 6] >= 0.3) == (episode.actions[:, 6] >= 0.3)
        ))
        uniform_binary_matches += int(np.count_nonzero(
            (uniform_decoded[:, 6] >= 0.3) == (episode.actions[:, 6] >= 0.3)
        ))
        gripper_count += len(episode.actions)
    return (
        adaptive_errors,
        uniform_errors,
        adaptive_binary_matches / gripper_count,
        uniform_binary_matches / gripper_count,
        np.asarray(selected_controls),
        np.asarray(maximum_controls),
    )


def compare(source: Path, adaptive: Path, uniform: Path, horizon: int) -> dict:
    episodes, _ = load_lerobot_episodes(source, ActionAlignment())
    adaptive_encoder = load_encoder(adaptive / "encoder.json")
    if tuple(adaptive_encoder.config.knot_insertion_excluded_dimensions) != (6,):
        raise ValueError("adaptive dataset must exclude only gripper dimension 6")

    adaptive_fixed_errors, adaptive_fixed_binary, padded = _stored_errors(
        episodes, adaptive, horizon
    )
    uniform_fixed_errors, uniform_fixed_binary, _ = _stored_errors(
        episodes, uniform, horizon
    )
    minimum_native_horizon = min(
        int(np.min(np.load(
            adaptive / "data" / f"episode_{episode.episode_index:06d}.npz"
        )["executable_steps"]))
        for episode in episodes
    )
    adaptive_safe_errors, adaptive_safe_binary, safe_padded = _stored_errors(
        episodes, adaptive, minimum_native_horizon
    )
    uniform_safe_errors, uniform_safe_binary, _ = _stored_errors(
        episodes, uniform, minimum_native_horizon
    )
    (
        adaptive_global_errors,
        uniform_global_errors,
        adaptive_global_binary,
        uniform_global_binary,
        excluded_selected,
        excluded_maximum,
    ) = _global_equal_budget_errors(episodes, adaptive_encoder)
    all_dim_encoder = create_encoder(replace(
        adaptive_encoder.config,
        knot_insertion_excluded_dimensions=(),
    ))
    (
        all_dim_global_errors,
        _,
        all_dim_global_binary,
        _,
        selected,
        maximum,
    ) = _global_equal_budget_errors(episodes, all_dim_encoder)

    adaptive_metrics = _metrics(adaptive_global_errors, 0.3)
    uniform_metrics = _metrics(uniform_global_errors, 0.3)
    all_dim_metrics = _metrics(all_dim_global_errors, 0.3)
    arm_improvement = 1.0 - adaptive_metrics["arm_rmse"] / uniform_metrics["arm_rmse"]
    adaptive_safe_metrics = _metrics(adaptive_safe_errors, 0.3)
    uniform_safe_metrics = _metrics(uniform_safe_errors, 0.3)
    safe_arm_improvement = (
        1.0 - adaptive_safe_metrics["arm_rmse"] / uniform_safe_metrics["arm_rmse"]
    )
    same_budget = bool(
        np.array_equal(excluded_selected, excluded_maximum)
        and np.array_equal(selected, maximum)
        and np.array_equal(selected, excluded_selected)
    )
    if not same_budget:
        raise ValueError("adaptive encoders did not consume the configured full control budget")
    if arm_improvement <= 0:
        raise ValueError("gripper-excluded adaptive representation did not beat uniform-left on arm RMSE")
    if safe_padded or safe_arm_improvement <= 0:
        raise ValueError("adaptive representation did not beat uniform-left on its guaranteed fixed prefix")

    return {
        "passed": True,
        "criterion": "adaptive-left arm RMSE must beat uniform-left at equal global control budget",
        "source_dataset": str(source),
        "fixed_horizon": horizon,
        "global_episode_fit_at_equal_control_budget": {
            "adaptive_excluded_gripper": {
                **adaptive_metrics,
                "binary_gripper_state_accuracy": adaptive_global_binary,
            },
            "uniform_knots": {
                **uniform_metrics,
                "binary_gripper_state_accuracy": uniform_global_binary,
            },
            "adaptive_all_dimensions_reference": {
                **all_dim_metrics,
                "binary_gripper_state_accuracy": all_dim_global_binary,
            },
        },
        "fixed_20_step_deployment_compatibility": {
            "adaptive_excluded_gripper_with_final_sample_padding": {
                **_metrics(adaptive_fixed_errors, 0.3),
                "binary_gripper_state_accuracy": adaptive_fixed_binary,
                "records_requiring_final_sample_padding": padded,
            },
            "uniform_left": {
                **_metrics(uniform_fixed_errors, 0.3),
                "binary_gripper_state_accuracy": uniform_fixed_binary,
            },
            "warning": (
                "Adaptive duration geometry has a variable native horizon. Padding preserves "
                "the fixed server tensor shape but is not equivalent to representing all 20 future steps."
            ),
        },
        "guaranteed_fixed_prefix": {
            "horizon_steps": minimum_native_horizon,
            "adaptive_excluded_gripper": {
                **adaptive_safe_metrics,
                "binary_gripper_state_accuracy": adaptive_safe_binary,
            },
            "uniform_left": {
                **uniform_safe_metrics,
                "binary_gripper_state_accuracy": uniform_safe_binary,
            },
            "adaptive_arm_rmse_reduction_vs_uniform_fraction": safe_arm_improvement,
            "adaptive_arm_rmse_reduction_vs_uniform_percent": 100.0 * safe_arm_improvement,
            "records_requiring_padding": safe_padded,
        },
        "equal_global_control_budget": {
            "verified": same_budget,
            "episode_count": len(episodes),
            "total_controls": int(excluded_selected.sum()),
            "per_episode_min": int(excluded_selected.min()),
            "per_episode_max": int(excluded_selected.max()),
            "all_episodes_hit_configured_cap": True,
        },
        "adaptive_arm_rmse_reduction_vs_uniform_fraction": arm_improvement,
        "adaptive_arm_rmse_reduction_vs_uniform_percent": 100.0 * arm_improvement,
        "note": (
            "Gripper dimension 6 does not place knots, so arm geometry improves while continuous "
            "gripper reconstruction is intentionally a separate tradeoff; thresholded state accuracy is reported."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("adaptive", type=Path)
    parser.add_argument("uniform", type=Path)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")
    source = args.source.expanduser().resolve()
    adaptive = args.adaptive.expanduser().resolve()
    uniform = args.uniform.expanduser().resolve()
    report = compare(source, adaptive, uniform, args.horizon)
    output = args.output or adaptive / "representation_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "same_budget": report["equal_global_control_budget"]["verified"],
        "arm_rmse_reduction_percent": report["adaptive_arm_rmse_reduction_vs_uniform_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
