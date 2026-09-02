#!/usr/bin/env python3
"""Compare SplineEncoder with dRTCv2 and its complete cached token records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np


MODES = ("uniform_double", "uniform_left", "adaptive_left")


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--drtcv2-root", type=Path, required=True)
    output.add_argument("--dataset", type=Path, required=True)
    output.add_argument("--report", type=Path)
    output.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return output


def padded_window(actions: np.ndarray, observation: int, count: int) -> np.ndarray:
    chunk = actions[observation : observation + count]
    if len(chunk) < count:
        chunk = np.concatenate((
            chunk,
            np.repeat(chunk[-1:], count - len(chunk), axis=0),
        ))
    return chunk


def cache_path(dataset: Path, mode: str) -> Path:
    pointer = (
        dataset / "action_tokenization_visualizations" / mode / "cache_path.txt"
    )
    path = Path(pointer.read_text().strip())
    return path if path.is_absolute() else dataset / path


def maximum(current: float, values) -> float:
    array = np.asarray(values)
    return max(current, float(np.max(np.abs(array))) if array.size else 0.0)


def verify_uniform(mode, dataset, episodes, new_load, old_load):
    artifact = (
        dataset / "action_tokenization_visualizations" / mode / "action_tokenizer.json"
    )
    new = new_load(artifact)
    old = old_load(artifact)
    root = cache_path(dataset, mode)
    metrics = {
        "mode": mode,
        "tokenizer_id": new.tokenizer_id,
        "original_tokenizer_id": old.tokenizer_id,
        "records": 0,
        "continuous_control_max_abs_difference": 0.0,
        "knot_max_abs_difference": float(np.max(np.abs(
            new.knots / new.config.frequency_hz
            - old.knots / old.config.frequency_hz
        ))),
        "dequantized_decode_max_abs_difference": 0.0,
        "new_vs_original_token_mismatches": 0,
        "new_vs_cache_token_mismatches": 0,
        "cache_key_mismatches": 0,
        "cache_mask_false_count": 0,
    }
    with h5py.File(root / "tokens.h5", "r") as handle:
        cached = handle["records"]
        cursor = 0
        for episode in episodes:
            for observation in range(len(episode.actions)):
                chunk = padded_window(
                    episode.actions, observation, new.config.input_chunk_size
                )
                new_result = new.encode_chunk(chunk, quantize=True)
                old_controls = old.fit_control_points(chunk)
                old_record = old.encode(chunk)
                metrics["continuous_control_max_abs_difference"] = maximum(
                    metrics["continuous_control_max_abs_difference"],
                    new_result.control_points - old_controls,
                )
                metrics["new_vs_original_token_mismatches"] += int(np.count_nonzero(
                    new_result.tokens != old_record.tokens
                ))
                metrics["dequantized_decode_max_abs_difference"] = maximum(
                    metrics["dequantized_decode_max_abs_difference"],
                    new_result.decode(dequantized=True) - old.decode(old_record.tokens),
                )
                metrics["new_vs_cache_token_mismatches"] += int(np.count_nonzero(
                    new_result.tokens != cached["action_tokens"][cursor]
                ))
                expected_key = np.uint64(
                    (int(episode.episode_index) << 32) | observation
                )
                metrics["cache_key_mismatches"] += int(
                    cached["key"][cursor] != expected_key
                )
                metrics["cache_mask_false_count"] += int(np.count_nonzero(
                    ~cached["action_token_mask"][cursor]
                ))
                cursor += 1
        metrics["records"] = cursor
        metrics["cache_records"] = int(len(cached["key"]))
    metrics["passed"] = bool(
        metrics["tokenizer_id"] == metrics["original_tokenizer_id"]
        and metrics["records"] == metrics["cache_records"]
        and metrics["continuous_control_max_abs_difference"] <= 1e-12
        and metrics["knot_max_abs_difference"] <= 1e-15
        and metrics["dequantized_decode_max_abs_difference"] <= 1e-12
        and metrics["new_vs_original_token_mismatches"] == 0
        and metrics["new_vs_cache_token_mismatches"] == 0
        and metrics["cache_key_mismatches"] == 0
        and metrics["cache_mask_false_count"] == 0
    )
    return metrics


def verify_adaptive(dataset, episodes, new_load, old_load):
    mode = "adaptive_left"
    artifact = (
        dataset / "action_tokenization_visualizations" / mode / "action_tokenizer.json"
    )
    new = new_load(artifact)
    old = old_load(artifact)
    root = cache_path(dataset, mode)
    metrics = {
        "mode": mode,
        "tokenizer_id": new.tokenizer_id,
        "original_tokenizer_id": old.tokenizer_id,
        "records": 0,
        "continuous_control_max_abs_difference": 0.0,
        "knot_max_abs_difference": 0.0,
        "dequantized_decode_max_abs_difference": 0.0,
        "duration_mismatches": 0,
        "new_vs_original_token_mismatches": 0,
        "new_vs_cache_token_mismatches": 0,
        "new_vs_cache_duration_mismatches": 0,
        "cache_key_mismatches": 0,
        "cache_mask_false_count": 0,
    }
    with h5py.File(root / "tokens.h5", "r") as handle:
        cached = handle["records"]
        cursor = 0
        for episode in episodes:
            new_records = new.encode_episode(episode.actions, quantize=True)
            old_parameters = old.episode_control_points(episode.actions)
            if len(new_records) != len(old_parameters):
                raise AssertionError(
                    f"adaptive episode {episode.episode_index} record count differs"
                )
            for new_result, (observation, old_controls, old_durations) in zip(
                new_records, old_parameters
            ):
                old_tokens = old._quantize(old_controls, old.config.vocab_size)
                old_geometry = old.geometry(old_durations)
                old_knots = old_geometry.knot_steps * old.config.delta_t
                metrics["continuous_control_max_abs_difference"] = maximum(
                    metrics["continuous_control_max_abs_difference"],
                    new_result.control_points - old_controls,
                )
                metrics["knot_max_abs_difference"] = maximum(
                    metrics["knot_max_abs_difference"], new_result.knots - old_knots
                )
                metrics["duration_mismatches"] += int(np.count_nonzero(
                    new_result.duration_steps != old_durations
                ))
                metrics["new_vs_original_token_mismatches"] += int(np.count_nonzero(
                    new_result.tokens != old_tokens
                ))
                metrics["dequantized_decode_max_abs_difference"] = maximum(
                    metrics["dequantized_decode_max_abs_difference"],
                    new_result.decode(dequantized=True)
                    - old.decode(old_tokens, old_durations),
                )
                metrics["new_vs_cache_token_mismatches"] += int(np.count_nonzero(
                    new_result.tokens != cached["action_tokens"][cursor]
                ))
                metrics["new_vs_cache_duration_mismatches"] += int(np.count_nonzero(
                    new_result.duration_steps
                    != cached["action_duration_tokens"][cursor]
                ))
                expected_key = np.uint64(
                    (int(episode.episode_index) << 32) | int(observation)
                )
                metrics["cache_key_mismatches"] += int(
                    cached["key"][cursor] != expected_key
                )
                metrics["cache_mask_false_count"] += int(np.count_nonzero(
                    ~cached["action_token_mask"][cursor]
                ))
                cursor += 1
        metrics["records"] = cursor
        metrics["cache_records"] = int(len(cached["key"]))
    metrics["passed"] = bool(
        metrics["tokenizer_id"] == metrics["original_tokenizer_id"]
        and metrics["records"] == metrics["cache_records"]
        and metrics["continuous_control_max_abs_difference"] <= 1e-12
        and metrics["knot_max_abs_difference"] <= 1e-15
        and metrics["dequantized_decode_max_abs_difference"] <= 1e-12
        and metrics["duration_mismatches"] == 0
        and metrics["new_vs_original_token_mismatches"] == 0
        and metrics["new_vs_cache_token_mismatches"] == 0
        and metrics["new_vs_cache_duration_mismatches"] == 0
        and metrics["cache_key_mismatches"] == 0
        and metrics["cache_mask_false_count"] == 0
    )
    return metrics


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository / "src"))
    sys.path.insert(1, str(args.drtcv2_root.resolve()))
    from spline_encoder import (
        infer_action_bounds,
        load_encoder,
        load_lerobot_episodes,
    )
    from tokenizer import load_tokenizer

    dataset = args.dataset.resolve()
    started = time.time()
    alignment = infer_action_bounds(dataset)
    episodes, info = load_lerobot_episodes(dataset, alignment)
    results = []
    for mode in args.modes:
        print(f"verifying {mode} across {sum(len(x.actions) for x in episodes)} observations",
              flush=True)
        if mode == "adaptive_left":
            result = verify_adaptive(dataset, episodes, load_encoder, load_tokenizer)
        else:
            result = verify_uniform(
                mode, dataset, episodes, load_encoder, load_tokenizer
            )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "schema_version": 1,
        "dataset": str(dataset),
        "dataset_codebase_version": info["codebase_version"],
        "dataset_fps": info["fps"],
        "episodes": len(episodes),
        "observations": int(sum(len(x.actions) for x in episodes)),
        "normalization": alignment.to_dict(),
        "modes": results,
        "passed": all(item["passed"] for item in results),
        "elapsed_seconds": time.time() - started,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "passed": report["passed"],
        "modes": args.modes,
        "elapsed_seconds": report["elapsed_seconds"],
        "report": None if args.report is None else str(args.report),
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
