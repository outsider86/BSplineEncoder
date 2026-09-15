"""Command-line LeRobot action-to-spline encoding."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import (
    ENCODER_MODES,
    AdaptiveLeftBSplineConfig,
    UniformBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    preset,
)
from .encoder import create_encoder
from .lerobot import ActionAlignment, encode_lerobot_dataset, infer_action_bounds
from .serialization import load_encoder


def _indices(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one action index is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="local LeRobot dataset root")
    parser.add_argument("output", type=Path, help="new encoded output directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--encoder-state", type=Path,
                        help="SplineEncoder or legacy action_tokenizer.json")
    source.add_argument("--preset", choices=("libero", "libero_chunk8", "robotwin",
                                             "piper_chunk50"))
    parser.add_argument("--mode", choices=ENCODER_MODES, default="uniform_left")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-indices", type=_indices,
                        help="selection/reordering such as 0,1,2,3,4,5,6")
    parser.add_argument("--normalization", choices=("none", "q01_q99"), default="none")
    parser.add_argument("--quantize", action="store_true",
                        help="also emit discrete control tokens")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrate token bounds on this dataset")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--frequency-hz", type=float, default=20.0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--num-basis", type=int, default=11)
    parser.add_argument("--span-length-steps", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--fit-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--exclude-knot-dimensions",
        type=_indices,
        help="adaptive-only dimensions excluded from knot insertion, e.g. 6 for a gripper",
    )
    return parser


def _custom_config(args):
    action_dim = args.action_dim
    if action_dim is None and args.action_indices is not None:
        action_dim = len(args.action_indices)
    if action_dim is None:
        raise SystemExit("--action-dim is required without --preset/--encoder-state")
    common = dict(
        action_dim=action_dim,
        chunk_size=args.chunk_size,
        frequency_hz=args.frequency_hz,
        degree=args.degree,
        num_basis=args.num_basis,
        span_length_steps=args.span_length_steps,
        vocab_size=args.vocab_size,
    )
    if args.mode == "uniform":
        return UniformBSplineConfig(**common, regularization=args.regularization)
    if args.mode == "uniform_double":
        return UniformDoubleBSplineConfig(**common, regularization=args.regularization)
    if args.mode == "uniform_left":
        return UniformLeftBSplineConfig(**common, regularization=args.regularization)
    return AdaptiveLeftBSplineConfig(
        **common,
        fit_tolerance=args.fit_tolerance,
        knot_insertion_excluded_dimensions=args.exclude_knot_dimensions or (),
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.encoder_state:
        if args.exclude_knot_dimensions is not None:
            raise SystemExit(
                "--exclude-knot-dimensions cannot override a serialized encoder state"
            )
        encoder = load_encoder(args.encoder_state)
    elif args.preset:
        config = preset(args.preset, args.mode)
        if args.exclude_knot_dimensions is not None:
            if args.mode not in {"adaptive", "adaptive_left"}:
                raise SystemExit("--exclude-knot-dimensions requires an adaptive mode")
            config = replace(
                config,
                knot_insertion_excluded_dimensions=args.exclude_knot_dimensions,
            )
        encoder = create_encoder(config)
    else:
        encoder = create_encoder(_custom_config(args))
    base_alignment = ActionAlignment(
        action_key=args.action_key,
        action_indices=args.action_indices,
    )
    alignment = (
        infer_action_bounds(
            args.dataset,
            action_key=args.action_key,
            action_indices=args.action_indices,
        )
        if args.normalization == "q01_q99"
        else base_alignment
    )
    result = encode_lerobot_dataset(
        args.dataset,
        args.output,
        encoder,
        alignment=alignment,
        quantize=args.quantize,
        calibrate_quantizer=args.calibrate,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "output": str(result.output_dir),
        "episodes": result.episode_count,
        "records": result.record_count,
        "quantized": result.quantized,
        "tokenizer_id": result.tokenizer_id,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
