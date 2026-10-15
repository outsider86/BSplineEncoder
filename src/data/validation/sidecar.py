"""Decode-and-compare verification of spline sidecars against cleaned actions.

Moved from the sidecar builder so all independent checks live in
``data.validation``.  Reads only the sidecar files and the dataset episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import BSpline


def _stats(errors: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(errors)
    return {
        "max_absolute_error": float(absolute.max()), "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "max_absolute_error_per_dimension": absolute.max(axis=0).tolist(),
    }


def verify_sidecar(dataset: Path, sidecar: Path, episodes: Iterable, sample_period: float,
                   quaternion_columns: tuple[int, int] | None = None,
                   train: set[int] | None = None) -> dict[str, Any]:
    """Decode every record and compare with the (aligned) cleaned actions.

    The reference for observation ``o`` is ``actions[o:o+steps]`` padded with
    the final action, which is the tail policy used by both uniform windows
    and the adaptive repeated-tail refit.
    """

    manifest = json.loads((Path(sidecar) / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((Path(sidecar) / "encoder.json").read_text())
    degree = state["config"]["degree"]
    calibration = state.get("calibration")
    errors, dequantized_errors, norms = [], [], []
    split_errors: dict[str, list] = {"train": [], "held_out": []}
    clipped = {"train": 0, "held_out": 0}
    flips = 0
    records = 0
    for episode in episodes:
        split = "train" if train is None or episode.episode_index in train else "held_out"
        path = Path(sidecar) / manifest["episode_path"].format(episode_index=episode.episode_index)
        with np.load(path) as stored:
            data = {key: stored[key] for key in stored.files}
        actions = episode.actions
        if not np.array_equal(data["frame_index"], episode.frame_indices):
            raise AssertionError(f"{path}: frame_index mapping differs from the dataset")
        if not np.array_equal(data["observation_index"], np.arange(len(actions))):
            raise AssertionError(f"{path}: not one record per observation")
        if len(data["controls"]) != len(actions):
            raise AssertionError(f"{path}: record count differs from dataset rows")
        for name in ("controls", "knots"):
            if not np.all(np.isfinite(data[name])):
                raise AssertionError(f"{path}: non-finite {name}")
        if calibration is not None:
            low, high = np.asarray(calibration["low"]), np.asarray(calibration["high"])
            controls = data["controls"]
            clipped[split] += int(np.count_nonzero((controls < low) | (controls > high)))
        if quaternion_columns is not None:
            start, end = quaternion_columns
            q = actions[:, start:end]
            flips += int(np.count_nonzero(np.einsum("ij,ij->i", q[1:], q[:-1]) < 0))
        first = len(errors)
        for observation in range(len(actions)):
            steps = int(data["executable_steps"][observation])
            reference = actions[observation: observation + steps]
            if len(reference) < steps:
                reference = np.concatenate(
                    (reference, np.repeat(actions[-1:], steps - len(reference), axis=0)))
            times = np.arange(steps) * sample_period
            knots = data["knots"][observation]
            decoded = BSpline(knots, data["controls"][observation], degree,
                              extrapolate=False)(times)
            if not np.all(np.isfinite(decoded)):
                raise AssertionError(f"{path}: observation {observation} decodes to NaN/Inf")
            errors.append(decoded - reference)
            if "dequantized_controls" in data:
                restored = BSpline(knots, data["dequantized_controls"][observation], degree,
                                   extrapolate=False)(times)
                dequantized_errors.append(restored - reference)
            if quaternion_columns is not None:
                start, end = quaternion_columns
                norms.append(np.linalg.norm(decoded[:, start:end], axis=1))
            records += 1
        split_errors[split].extend(errors[first:])
    report: dict[str, Any] = {
        "records": records,
        "decoded_rows": int(sum(len(e) for e in errors)),
        "all_finite": True,
        "fitting_error": _stats(np.concatenate(errors)),
        "reference": "cleaned dataset actions (after optional sidecar normalization), "
                     "tail padded with the final action",
    }
    report["fitting_error_by_split"] = {
        name: _stats(np.concatenate(values)) for name, values in split_errors.items() if values}
    if dequantized_errors:
        report["dequantized_error"] = _stats(np.concatenate(dequantized_errors))
    if calibration is not None:
        report["controls_outside_calibration"] = {
            **clipped, "note": "clipped to calibration bounds when quantized; held-out "
                               "episodes are not part of the training-only calibration"}
    if norms:
        values = np.concatenate(norms)
        report["decoded_quaternion_norm"] = {
            "min": float(values.min()), "max": float(values.max()),
            "max_abs_deviation_from_unit": float(np.abs(values - 1).max()),
            "note": "quaternion components are fitted as independent scalars; "
                    "renormalize decoded quaternions",
        }
        report["reference_quaternion_sign_flips"] = flips
        if flips:
            report["warning"] = (
                f"{flips} consecutive action quaternions have negative dot products (q/-q "
                "sign flips); scalar splines and quantization see large jumps. Export with "
                "quaternion='continuous' for spline sidecars of absolute EE actions.")
    return report


__all__ = ["verify_sidecar"]
