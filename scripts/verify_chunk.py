#!/usr/bin/env python3
"""Verify and optionally visualize one NumPy action window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from spline_encoder import (
    ENCODER_MODES,
    create_encoder,
    load_encoder,
    preset,
    save_reconstruction_plot,
    verify_reconstruction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", type=Path, help=".npy array with shape (steps, action_dim)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--encoder-state", type=Path)
    source.add_argument(
        "--preset",
        choices=("libero", "libero_chunk8", "robotwin", "piper_chunk50"),
    )
    parser.add_argument("--mode", choices=ENCODER_MODES, default="uniform_left")
    parser.add_argument("--max-error", type=float)
    parser.add_argument("--dequantized", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--dimensions", help="comma-separated action dimensions to plot")
    return parser


def _dimensions(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("dimensions must be comma-separated integers") from exc
    if not result:
        raise ValueError("at least one plot dimension is required")
    return result


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    encoder = (
        load_encoder(args.encoder_state)
        if args.encoder_state is not None
        else create_encoder(preset(args.preset, args.mode))
    )
    actions = np.asarray(np.load(args.actions, allow_pickle=False), dtype=np.float64)
    expected = (encoder.config.input_chunk_size, encoder.config.action_dim)
    if actions.shape != expected:
        raise ValueError(f"action window must have shape {expected}, got {actions.shape}")
    if args.dequantized and not encoder.calibrated:
        raise ValueError("--dequantized requires a calibrated --encoder-state")
    result = encoder.encode_chunk(actions, quantize=args.dequantized)
    reference = actions[: result.executable_steps]
    if len(reference) < result.executable_steps:
        reference = np.concatenate((
            reference,
            np.repeat(reference[-1:], result.executable_steps - len(reference), axis=0),
        ))
    report = verify_reconstruction(
        result,
        reference,
        max_absolute_error=args.max_error,
        dequantized=args.dequantized,
    )
    report.update({
        "actions": str(args.actions.expanduser().resolve()),
        "mode": encoder.config.mode,
        "tokenizer_id": encoder.tokenizer_id,
    })
    if args.plot is not None:
        try:
            dimensions = _dimensions(args.dimensions)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        save_reconstruction_plot(
            result,
            args.plot,
            reference,
            dimensions=dimensions,
        )
        report["plot"] = str(args.plot.expanduser().resolve())
    if args.report is not None:
        report_path = args.report.expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
