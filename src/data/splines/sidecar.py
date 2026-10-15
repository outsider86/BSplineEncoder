"""Optional B-spline sidecars over the FINAL cleaned ``action`` column.

The exported LeRobot actions are encoded exactly as stored: no differencing,
shifting, or (unless ``normalization="q01_q99"`` is requested once)
normalization is applied here.  ``action_dim`` and ``frequency_hz`` are taken
from the dataset; incompatible presets or saved encoders are rejected.  The
sidecar schema is the unchanged :func:`spline_encoder.encode_lerobot_dataset`
output plus one additive ``preprocessing_provenance.json``.

Scalar spline fitting treats every action dimension independently.  For
absolute-EE actions the four quaternion components are fitted as scalars, so
decoded quaternions are generally NOT unit length; renormalize after
decoding.  The verification report measures that deviation.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from spline_encoder.encoding.config import is_adaptive_mode, preset as spline_preset
from spline_encoder.encoding.encoder import create_encoder
from spline_encoder.encoding.serialization import load_encoder

from ..config import SplineConfig
from ..validation.sidecar import verify_sidecar
from .cli import _custom_config
from .lerobot import (
    ActionAlignment, calibrate_from_lerobot, encode_lerobot_dataset, load_lerobot_episodes,
)


def dataset_geometry(dataset: Path) -> tuple[int, float]:
    info = json.loads((Path(dataset) / "meta/info.json").read_text(encoding="utf-8"))
    shape = info["features"]["action"]["shape"]
    if len(shape) != 1 or shape[0] < 1:
        raise ValueError(f"{dataset}: action feature must be one-dimensional, got {shape}")
    return int(shape[0]), float(info["fps"])


def build_encoder(spline: SplineConfig, action_dim: int, fps: float):
    """Create (or load) an encoder whose geometry matches the dataset."""

    spline.check()
    if spline.encoder_state is not None:
        encoder = load_encoder(Path(spline.encoder_state).expanduser())
        config = encoder.config
        source = f"encoder_state {spline.encoder_state}"
    elif spline.preset is not None:
        config = spline_preset(spline.preset, spline.mode)
        source = f"preset {spline.preset}"
        encoder = None
    else:
        namespace = type("Args", (), dict(
            action_dim=action_dim, action_indices=None, chunk_size=spline.chunk_size,
            frequency_hz=fps, degree=spline.degree, num_basis=spline.num_basis,
            span_length_steps=spline.span_length_steps, vocab_size=spline.vocab_size,
            regularization=spline.regularization, fit_tolerance=spline.fit_tolerance,
            exclude_knot_dimensions=tuple(spline.exclude_knot_dimensions) or None,
            mode=spline.mode,
        ))
        if spline.exclude_knot_dimensions and not is_adaptive_mode(spline.mode):
            raise ValueError("exclude_knot_dimensions requires an adaptive spline mode")
        config = _custom_config(namespace)
        source = "custom geometry"
        encoder = None
    if config.action_dim != action_dim:
        raise ValueError(f"{source}: action_dim {config.action_dim} != dataset action_dim "
                         f"{action_dim}")
    if not np.isclose(config.frequency_hz, fps):
        raise ValueError(f"{source}: frequency_hz {config.frequency_hz} != dataset fps {fps}")
    if spline.exclude_knot_dimensions and spline.preset is not None:
        if not is_adaptive_mode(config.mode):
            raise ValueError("exclude_knot_dimensions requires an adaptive spline mode")
        config = replace(config, knot_insertion_excluded_dimensions=tuple(
            spline.exclude_knot_dimensions))
    return encoder if encoder is not None else create_encoder(config)


def train_episodes(total: int, validation: int) -> list[int]:
    if validation >= total:
        raise ValueError(f"validation_episodes={validation} leaves no training episodes")
    return list(range(total - validation))


def _quantile_alignment(episodes, train: set[int]) -> ActionAlignment:
    actions = np.concatenate([e.actions for e in episodes if e.episode_index in train], axis=0)
    return ActionAlignment(low=tuple(np.quantile(actions, 0.01, axis=0).tolist()),
                           high=tuple(np.quantile(actions, 0.99, axis=0).tolist()), clip=True)


def dataset_fingerprint(dataset: Path) -> str:
    digest = sha256()
    for path in sorted([Path(dataset) / "meta/info.json",
                        *(Path(dataset) / "data").rglob("*.parquet")]):
        digest.update(path.relative_to(dataset).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def encode_sidecar(dataset: Path, output: Path, spline: SplineConfig, *,
                   final_dataset: Path, quaternion_columns: tuple[int, int] | None,
                   provenance: dict[str, Any]) -> dict[str, Any]:
    """Encode ``dataset`` (possibly a staging copy) into a new ``output``."""

    action_dim, fps = dataset_geometry(dataset)
    encoder = build_encoder(spline, action_dim, fps)
    base = ActionAlignment()
    episodes, _ = load_lerobot_episodes(dataset, base)
    train = set(train_episodes(len(episodes), spline.validation_episodes))
    alignment = base
    if spline.normalization == "q01_q99":
        alignment = _quantile_alignment(episodes, train)
        episodes, _ = load_lerobot_episodes(dataset, alignment)
    calibration = "none"
    if spline.quantize:
        if encoder.calibrated and spline.encoder_state is not None:
            calibration = f"saved calibration from {spline.encoder_state}"
        else:
            calibrate_from_lerobot(encoder, [e for e in episodes if e.episode_index in train])
            calibration = f"training episodes {min(train)}..{max(train)}"
    result = encode_lerobot_dataset(dataset, output, encoder, alignment=alignment,
                                    quantize=spline.quantize, calibrate_quantizer=False)
    manifest_path = Path(output) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dataset"] = str(Path(final_dataset).resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verification = verify_sidecar(dataset, output, episodes, 1.0 / fps, quaternion_columns, train)
    tail = ("uniform windows repeat the final action"
            if not is_adaptive_mode(encoder.config.mode)
            else "adaptive refit pads the episode end with repeated final actions")
    record = {
        "schema_version": 1,
        "dataset": str(Path(final_dataset).resolve()),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "action_dim": action_dim, "fps": fps,
        "spline": json.loads(json.dumps(spline.__dict__, default=list)),
        "tokenizer_id": encoder.tokenizer_id, "mode": encoder.config.mode,
        "tail_policy": tail,
        "train_episodes": sorted(train),
        "normalization": alignment.to_dict(),
        "quantization_calibration": calibration,
        "action_transforms_in_sidecar": (["q01_q99 normalization from training episodes"]
                                         if spline.normalization == "q01_q99" else []),
        "verification": verification,
        **provenance,
    }
    (Path(output) / "preprocessing_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"records": result.record_count, "tokenizer_id": result.tokenizer_id,
            "verification": verification}


__all__ = ["build_encoder", "dataset_geometry", "encode_sidecar", "verify_sidecar"]  # re-export
