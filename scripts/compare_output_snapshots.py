#!/usr/bin/env python3
"""Snapshot deterministic encoder outputs and compare two package versions.

Usage (CPU only, a few seconds):

    git archive <commit> src | tar -x -C /tmp/baseline
    python scripts/compare_output_snapshots.py snapshot --src /tmp/baseline/src --output base.npz
    python scripts/compare_output_snapshots.py snapshot --src src --output new.npz
    python scripts/compare_output_snapshots.py compare base.npz new.npz

``compare`` requires bit-identical arrays and reports every differing key.
Only behavior that is supposed to be unchanged by the 2026-10 review is
snapshotted; intentionally corrected outputs are covered by regression tests.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


def _episode(length: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    smooth = np.stack([
        np.sin(2 * np.pi * rng.uniform(0.3, 1.2) * (i + 1) * t + rng.uniform(0, 3))
        for i in range(dim)
    ], axis=1)
    return smooth + 0.01 * rng.normal(size=smooth.shape)


def snapshot(src: Path) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(src.resolve()))
    import spline_encoder as se
    from spline_encoder.subdivision import extract_left_clamped_prefix

    out: dict[str, np.ndarray] = {}
    print(json.dumps({"imported": se.__file__}))
    configs = {}
    for name in ("libero", "libero_chunk8", "robotwin", "piper_chunk50"):
        for mode in se.ENCODER_MODES:
            configs[f"{name}/{mode}"] = se.preset(name, mode)
        configs[f"{name}/legacy_left"] = replace(
            se.preset(name, "uniform_left"),
            implementation_version="uniform_left_extended_double_fit_v1",
        )
    for degree in range(1, 6):
        configs[f"deg{degree}/inclusive"] = se.UniformDoubleBSplineConfig(
            action_dim=3, chunk_size=24, frequency_hz=30.0, degree=degree,
            num_basis=degree + 7, span_length_steps=None, regularization=0.0,
        )
        configs[f"deg{degree}/left"] = se.UniformLeftBSplineConfig(
            action_dim=3, chunk_size=24, frequency_hz=30.0, degree=degree,
            num_basis=degree + 12, span_length_steps=2,
        )
        # span_length_steps=None: average span (chunk - 1) / (num_basis - degree) = 2.
        configs[f"deg{degree}/adaptive"] = se.AdaptiveLeftBSplineConfig(
            action_dim=3, chunk_size=2 * (degree + 3) + 1, frequency_hz=30.0,
            degree=degree, num_basis=2 * degree + 3, span_length_steps=None,
        )
    configs["adaptive_excluded"] = replace(
        se.preset("libero_chunk8", "adaptive_left"), knot_insertion_excluded_dimensions=(6,)
    )

    for key, config in configs.items():
        encoder = se.create_encoder(config)
        out[f"{key}/fingerprint"] = np.frombuffer(config.fingerprint.encode(), dtype=np.uint8)
        dim = config.action_dim
        if se.is_adaptive_mode(config.mode):
            episode = _episode(70, dim, seed=len(key))
            encoder.calibrate([episode])
            records = encoder.encode_episode(episode, quantize=True)
            for name in ("control_points", "knots", "duration_steps", "tokens",
                         "dequantized_control_points"):
                out[f"{key}/{name}"] = np.stack([getattr(r, name) for r in records])
            out[f"{key}/decode"] = np.concatenate([r.decode() for r in records])
        else:
            count = config.input_chunk_size
            episode = _episode(count + 20, dim, seed=len(key))
            windows = [episode[i : i + count] for i in range(21)]
            encoder.calibrate(windows)
            results = [encoder.encode_chunk(w, quantize=True) for w in windows]
            for name in ("control_points", "knots", "tokens", "dequantized_control_points"):
                out[f"{key}/{name}"] = np.stack([getattr(r, name) for r in results])
            out[f"{key}/decode"] = np.stack([r.decode() for r in results])
            out[f"{key}/decode_dequantized"] = np.stack(
                [r.decode(dequantized=True) for r in results]
            )
            out[f"{key}/encoder_decode"] = encoder.decode(results[0].tokens)
            state = se.encoder_to_state(encoder)
            out[f"{key}/state"] = np.frombuffer(
                json.dumps(state, sort_keys=True).encode(), dtype=np.uint8
            )
            # Exclusive-layout subdivision and prefix extraction of clamped inputs.
            first = results[0]
            if not first.metadata.get("inclusive_domain", False):
                left, right = se.subdivide_parameters(first, first.executable_steps // 2)
                for side, part in (("left", left), ("right", right)):
                    out[f"{key}/subdivide_{side}_c"] = part.control_points
                    out[f"{key}/subdivide_{side}_t"] = part.knots
            spline = first.scipy_spline()
            start, end = float(spline.t[spline.k]), float(spline.t[len(spline.c)])
            if first.metadata.get("boundary_condition") == "double_clamped":
                for a, b in ((start, end), (0.3 * end, end), (0.2 * end, 0.7 * end)):
                    interval = se.extract_interval(spline, a, b)
                    out[f"{key}/interval_{a:.4f}_{b:.4f}_t"] = interval.t
                    out[f"{key}/interval_{a:.4f}_{b:.4f}_c"] = interval.c
            prefix = extract_left_clamped_prefix(spline, 0.0, num_knot_spans=spline.k + 1)
            out[f"{key}/prefix0_c"] = prefix.control_points
            out[f"{key}/prefix0_k"] = prefix.distinct_knots
    sys.path.pop(0)
    return out


def compare(first: Path, second: Path) -> int:
    with np.load(first) as a, np.load(second) as b:
        keys_a, keys_b = set(a.files), set(b.files)
        differing = sorted(
            key for key in keys_a & keys_b
            if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key])
        )
        report = {
            "compared_arrays": len(keys_a & keys_b),
            "compared_values": int(sum(a[key].size for key in keys_a & keys_b)),
            "only_in_first": sorted(keys_a - keys_b),
            "only_in_second": sorted(keys_b - keys_a),
            "bitwise_differences": differing,
        }
    report["identical"] = not (
        report["only_in_first"] or report["only_in_second"] or report["bitwise_differences"]
    )
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("snapshot")
    make.add_argument("--src", type=Path, required=True, help="directory containing spline_encoder/")
    make.add_argument("--output", type=Path, required=True)
    diff = commands.add_parser("compare")
    diff.add_argument("first", type=Path)
    diff.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare(args.first, args.second)
    values = snapshot(args.src)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **values)
    print(json.dumps({"arrays": len(values), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
