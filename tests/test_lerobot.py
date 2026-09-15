import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from spline_encoder import (
    ActionAlignment,
    AdaptiveLeftBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    create_encoder,
    encode_lerobot_dataset,
    infer_action_bounds,
    load_lerobot_episodes,
)


def make_dataset(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 10.0,
        "features": {
            "action": {"dtype": "float32", "shape": [4], "names": ["action"]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    for episode_index, length in [(0, 7), (1, 6)]:
        rows = []
        for frame in range(length):
            rows.append({
                "episode_index": episode_index,
                "frame_index": frame,
                "timestamp": frame / 10,
                "action": np.asarray([
                    episode_index + frame,
                    100 + frame,
                    -frame,
                    frame / 10,
                ], dtype=np.float32),
            })
        pd.DataFrame(rows).to_parquet(
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
    return root


def test_alignment_selection_reordering_and_quantile_bounds(tmp_path):
    root = make_dataset(tmp_path)
    alignment = ActionAlignment(action_indices=(2, 0))
    episodes, info = load_lerobot_episodes(root, alignment)
    assert info["fps"] == 10
    np.testing.assert_array_equal(episodes[0].actions[2], [-2, 2])
    normalized = infer_action_bounds(root, action_indices=(2, 0),
                                     lower_quantile=0, upper_quantile=1)
    normalized_episodes, _ = load_lerobot_episodes(root, normalized)
    assert np.min(np.concatenate([x.actions for x in normalized_episodes])) == -1
    assert np.max(np.concatenate([x.actions for x in normalized_episodes])) == 1


def test_alignment_rejects_nonintegral_indices_nonfinite_bounds_and_actions():
    with pytest.raises(ValueError, match="non-negative indices"):
        ActionAlignment(action_indices=(0, 1.5))
    with pytest.raises(ValueError, match="bounds"):
        ActionAlignment(low=(0.0, np.nan), high=(1.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        ActionAlignment().apply([[0.0, np.inf]])


def test_dataset_export_is_observation_aligned_and_continuous_first(tmp_path):
    root = make_dataset(tmp_path)
    config = UniformDoubleBSplineConfig(
        action_dim=2, chunk_size=6, frequency_hz=10,
        degree=2, num_basis=4, span_length_steps=3,
    )
    encoder = create_encoder(config)
    output = tmp_path / "encoded"
    result = encode_lerobot_dataset(
        root, output, encoder,
        alignment=ActionAlignment(action_indices=(2, 0)),
    )
    assert result.episode_count == 2
    assert result.record_count == 13
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["continuous_controls"] is True
    assert manifest["quantized"] is False
    with np.load(output / "data" / "episode_000000.npz") as data:
        assert data["controls"].shape == (7, 4, 2)
        assert data["knots"].shape == (7, 7)
        np.testing.assert_array_equal(data["observation_index"], np.arange(7))
        assert "tokens" not in data
        assert "duration_steps" not in data
        assert "control_duration_interleaved" not in data


def test_validation_cli_recomputes_uniform_episode_records(tmp_path):
    root = make_dataset(tmp_path)
    encoder = create_encoder(UniformLeftBSplineConfig(
        action_dim=2, chunk_size=6, frequency_hz=10,
        degree=2, num_basis=4, span_length_steps=3,
    ))
    output = tmp_path / "encoded"
    encode_lerobot_dataset(
        root,
        output,
        encoder,
        alignment=ActionAlignment(action_indices=(2, 0)),
    )
    report = tmp_path / "validation.json"
    repository = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "validate_lerobot_encoding.py"),
            str(root),
            str(output),
            "--output",
            str(report),
            "--fixed-horizon",
            "4",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    validation = json.loads(report.read_text())
    assert validation["passed"]
    assert validation["observations_checked"] == 13
    assert validation["full_recomputation_exact"]
    assert validation["interleaving"] is None
    assert validation["duration_steps"]["min"] == 3
    assert validation["duration_steps"]["max"] == 3


def test_quantized_dataset_export_adds_tokens(tmp_path):
    root = make_dataset(tmp_path)
    encoder = create_encoder(UniformDoubleBSplineConfig(
        action_dim=2, chunk_size=6, frequency_hz=10,
        degree=2, num_basis=4, span_length_steps=3,
    ))
    output = tmp_path / "encoded"
    encode_lerobot_dataset(
        root, output, encoder,
        alignment=ActionAlignment(action_indices=(2, 0)),
        quantize=True, calibrate_quantizer=True,
    )
    with np.load(output / "data" / "episode_000001.npz") as data:
        assert data["tokens"].shape == (6, 4, 2)
        assert data["dequantized_controls"].shape == (6, 4, 2)


def test_adaptive_export_contains_control_duration_interleaved_rows(tmp_path):
    root = make_dataset(tmp_path)
    encoder = create_encoder(AdaptiveLeftBSplineConfig(
        action_dim=2,
        chunk_size=6,
        frequency_hz=10,
        degree=2,
        num_basis=5,
        span_length_steps=2,
        knot_insertion_excluded_dimensions=(1,),
    ))
    output = tmp_path / "adaptive"
    encode_lerobot_dataset(
        root,
        output,
        encoder,
        alignment=ActionAlignment(action_indices=(2, 0)),
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["policy_layout"] == "control_point_duration_interleaved_v1"
    with np.load(output / "data" / "episode_000000.npz") as data:
        assert data["controls"].shape == (7, 5, 2)
        assert data["duration_steps"].shape == (7, 5)
        assert data["control_duration_interleaved"].shape == (7, 5, 3)
        np.testing.assert_array_equal(
            data["control_duration_interleaved"][:, :, :2], data["controls"]
        )
        np.testing.assert_array_equal(
            data["control_duration_interleaved"][:, :, 2], data["duration_steps"]
        )
