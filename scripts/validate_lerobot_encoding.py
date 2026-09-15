#!/usr/bin/env python3
"""Recompute and validate every record in an encoded LeRobot sidecar."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from spline_encoder import (
    ActionAlignment,
    iter_encoded_episodes,
    load_encoder,
    load_lerobot_episodes,
)
from spline_encoder.config import is_adaptive_mode


def _metrics(errors: list[np.ndarray]) -> dict:
    values = np.concatenate([item.reshape(-1, item.shape[-1]) for item in errors], axis=0)
    absolute = np.abs(values)
    return {
        "count": int(values.shape[0]),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error": float(absolute.max()),
        "mae_per_action_dimension": absolute.mean(axis=0).tolist(),
        "rmse_per_action_dimension": np.sqrt(np.mean(np.square(values), axis=0)).tolist(),
        "max_absolute_error_per_action_dimension": absolute.max(axis=0).tolist(),
    }


def validate(source: Path, encoded: Path, *, fixed_horizon: int = 20) -> dict:
    manifest = json.loads((encoded / "manifest.json").read_text())
    alignment = ActionAlignment(**{
        key: (tuple(value) if key in {"action_indices", "low", "high"} and value is not None else value)
        for key, value in manifest["alignment"].items()
    })
    episodes, info = load_lerobot_episodes(source, alignment)
    encoder = load_encoder(encoded / "encoder.json")
    errors: list[np.ndarray] = []
    fixed_horizon_errors: list[np.ndarray] = []
    durations = []
    executable_steps = []
    digest = sha256()
    observations = 0
    maximum = {"absolute_error": -1.0}
    fixed_horizon_padded_records = 0
    adaptive = is_adaptive_mode(encoder.config.mode)

    if manifest["tokenizer_id"] != encoder.tokenizer_id:
        raise ValueError("manifest and encoder tokenizer IDs differ")
    if manifest["source_fps"] != info["fps"]:
        raise ValueError("manifest and source FPS differ")

    for episode in episodes:
        path = encoded / "data" / f"episode_{episode.episode_index:06d}.npz"
        with np.load(path) as stored:
            payload = {key: stored[key] for key in stored.files}
        recomputed = next(iter_encoded_episodes(encoder, [episode])).splines
        if len(recomputed) != len(episode.actions):
            raise ValueError(f"episode {episode.episode_index} is not observation aligned")

        digest.update(episode.frame_indices.astype("<i8", copy=False).tobytes())
        digest.update(episode.actions.astype("<f8", copy=False).tobytes())
        np.testing.assert_array_equal(payload["frame_index"], episode.frame_indices)
        np.testing.assert_array_equal(payload["observation_index"], np.arange(len(recomputed)))
        np.testing.assert_array_equal(
            payload["controls"], np.stack([record.control_points for record in recomputed])
        )
        np.testing.assert_array_equal(
            payload["knots"], np.stack([record.knots for record in recomputed])
        )
        np.testing.assert_array_equal(
            payload["executable_steps"],
            np.asarray([record.executable_steps for record in recomputed]),
        )

        if recomputed[0].duration_steps is not None:
            expected_durations = np.stack([record.duration_steps for record in recomputed])
            np.testing.assert_array_equal(payload["duration_steps"], expected_durations)
        if adaptive:
            expected_interleaved = np.stack([
                record.interleaved_control_durations() for record in recomputed
            ])
            np.testing.assert_array_equal(
                payload["control_duration_interleaved"], expected_interleaved
            )
            np.testing.assert_array_equal(
                payload["control_duration_interleaved"][..., :-1], payload["controls"]
            )
            np.testing.assert_array_equal(
                payload["control_duration_interleaved"][..., -1], payload["duration_steps"]
            )
        elif "control_duration_interleaved" in payload:
            raise ValueError("uniform encoding unexpectedly contains adaptive interleaved rows")

        for record in recomputed:
            if record.observation_index is None:
                raise ValueError("record lacks observation_index")
            if record.duration_steps is not None:
                reconstructed_knots = np.r_[
                    np.zeros(record.degree + 1, dtype=np.int64),
                    np.cumsum(record.duration_steps),
                ]
                np.testing.assert_array_equal(
                    reconstructed_knots, np.rint(record.knot_steps).astype(np.int64)
                )
                durations.append(record.duration_steps)
            decoded = record.decode()
            start = int(record.observation_index)
            target_indices = np.minimum(
                start + np.arange(record.executable_steps), len(episode.actions) - 1
            )
            error = decoded - episode.actions[target_indices]
            errors.append(error)
            # Deployment needs a fixed action tensor. Adaptive splines can
            # occasionally have fewer native executable samples than that
            # tensor, so repeat the final valid sample rather than extrapolate.
            fixed_steps = np.minimum(
                np.arange(fixed_horizon, dtype=np.int64),
                record.executable_steps - 1,
            )
            fixed_decoded = np.asarray(
                record.scipy_spline()(fixed_steps * record.sample_period),
                dtype=np.float64,
            )
            fixed_target_indices = np.minimum(
                start + np.arange(fixed_horizon), len(episode.actions) - 1
            )
            fixed_horizon_errors.append(
                fixed_decoded - episode.actions[fixed_target_indices]
            )
            fixed_horizon_padded_records += int(record.executable_steps < fixed_horizon)
            local = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            local_max = float(np.abs(error[local]))
            if local_max > maximum["absolute_error"]:
                maximum = {
                    "absolute_error": local_max,
                    "episode_index": int(episode.episode_index),
                    "observation_index": start,
                    "horizon_step": int(local[0]),
                    "action_dimension": int(local[1]),
                    "target": float(episode.actions[target_indices][local]),
                    "decoded": float(decoded[local]),
                }
            executable_steps.append(record.executable_steps)
        observations += len(recomputed)

    if observations != manifest["record_count"]:
        raise ValueError("recomputed observation count differs from manifest")
    duration_values = np.concatenate(durations) if durations else np.empty(0)
    executable = np.asarray(executable_steps)
    result = {
        "passed": True,
        "source_dataset": str(source),
        "encoded_dataset": str(encoded),
        "source_fingerprint_matches_manifest": (
            digest.hexdigest() == manifest["source_action_fingerprint"]
        ),
        "episodes_checked": len(episodes),
        "observations_checked": observations,
        "tokenizer_id": encoder.tokenizer_id,
        "configuration": vars(encoder.config),
        "policy_layout": manifest.get("policy_layout"),
        "interleaving": ({
            "field": "control_duration_interleaved",
            "row_semantics": "[control_point_i[action dimensions...], duration_i]",
            "shape_per_observation": [
                encoder.config.num_basis, encoder.config.action_dim + 1
            ],
            "controls_round_trip_exact": True,
            "durations_round_trip_exact": True,
            "knots_reconstructed_from_durations_exact": True,
        } if adaptive else None),
        "duration_steps": ({
            "count": int(duration_values.size),
            "min": int(duration_values.min()),
            "median": float(np.median(duration_values)),
            "mean": float(duration_values.mean()),
            "p95": float(np.quantile(duration_values, 0.95)),
            "max": int(duration_values.max()),
        } if duration_values.size else None),
        "executable_horizon_steps": {
            "min": int(executable.min()),
            "median": float(np.median(executable)),
            "mean": float(executable.mean()),
            "p95": float(np.quantile(executable, 0.95)),
            "max": int(executable.max()),
        },
        "continuous_decode_metrics": _metrics(errors),
        "fixed_horizon_decode": {
            "horizon_steps": fixed_horizon,
            "padding": "repeat_final_native_sample",
            "padded_records": fixed_horizon_padded_records,
            "padded_record_fraction": fixed_horizon_padded_records / observations,
            "metrics": _metrics(fixed_horizon_errors),
        },
        "max_error_location": maximum,
        "full_recomputation_exact": True,
    }
    if not result["source_fingerprint_matches_manifest"]:
        raise ValueError("source action fingerprint differs from manifest")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("encoded", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixed-horizon", type=int, default=20)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    encoded = args.encoded.expanduser().resolve()
    output = args.output or encoded / "validation.json"
    if args.fixed_horizon < 1:
        parser.error("--fixed-horizon must be positive")
    report = validate(source, encoded, fixed_horizon=args.fixed_horizon)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "observations": report["observations_checked"],
        "rmse": report["continuous_decode_metrics"]["rmse"],
        "max_error": report["continuous_decode_metrics"]["max_absolute_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
