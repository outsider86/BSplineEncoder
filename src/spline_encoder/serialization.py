"""JSON serialization, including legacy dRTCv2 tokenizer artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .config import config_from_dict
from .encoder import create_encoder


ENCODER_STATE_SCHEMA_VERSION = 1


def encoder_to_state(encoder) -> dict[str, Any]:
    calibration = encoder.calibration_state() if encoder.calibrated else None
    return {
        "schema_version": ENCODER_STATE_SCHEMA_VERSION,
        "library": "spline_encoder",
        "tokenizer_id": encoder.config.fingerprint,
        "config": asdict(encoder.config),
        "calibration": calibration,
        "quantization_available": calibration is not None,
    }


def encoder_from_state(state: Mapping[str, Any]):
    payload = dict(state)
    library = payload.get("library")
    if library not in {"spline_encoder", "action_tokenization"}:
        raise ValueError(f"unexpected library identifier: {library!r}")
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("encoder state is missing a config mapping")
    config = config_from_dict(payload["config"])
    if payload.get("tokenizer_id") != config.fingerprint:
        raise ValueError("encoder fingerprint does not match its configuration")
    calibration = payload.get("calibration")
    if calibration is not None and not isinstance(calibration, Mapping):
        raise ValueError("calibration must be an object or null")
    # action_tokenization schema 1 and 2 use the same numerical calibration.
    if library == "action_tokenization" and payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported legacy tokenizer state schema")
    if library == "spline_encoder" and payload.get("schema_version") != 1:
        raise ValueError("unsupported SplineEncoder state schema")
    return create_encoder(config, None if calibration is None else dict(calibration))


def save_encoder(encoder, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(encoder_to_state(encoder), temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target


def load_encoder(path: str | Path):
    target = Path(path)
    try:
        payload = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read encoder state from {target}") from exc
    if not isinstance(payload, dict):
        raise ValueError("encoder state root must be a JSON object")
    return encoder_from_state(payload)


# Familiar aliases for migration from the original tokenizer module.
save_tokenizer = save_encoder
load_tokenizer = load_encoder


__all__ = [
    "ENCODER_STATE_SCHEMA_VERSION", "encoder_from_state", "encoder_to_state",
    "load_encoder", "load_tokenizer", "save_encoder", "save_tokenizer",
]
