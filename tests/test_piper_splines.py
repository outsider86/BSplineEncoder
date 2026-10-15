"""Spline sidecars produced by the Piper pipeline on the FINAL cleaned actions."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

from piper_fixtures import make_raw_episode  # noqa: E402
from spline_encoder import (  # noqa: E402
    UniformLeftBSplineConfig, create_encoder, iter_encoded_episodes, load_encoder,
    load_lerobot_episodes, save_encoder,
)
from data import PipelineConfig, Setup, SplineConfig, run  # noqa: E402
from data.splines.lerobot import ActionAlignment  # noqa: E402


@pytest.fixture(scope="module")
def raw_task(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw_spline") / "classify_blocks"
    make_raw_episode(root / "ep_a")
    make_raw_episode(root / "ep_b", onset=1.6)
    make_raw_episode(root / "ep_c", onset=0.9, pause=(2.0, 2.5))
    return root


def _run(raw_task, output_root, spline: SplineConfig, setups=("state-delta-joint",), **extra):
    config = PipelineConfig(raw_root=str(raw_task), output_root=str(output_root), setups=setups,
                            cameras=(), spline=spline, **extra)
    return run(config, log=lambda *_: None)[0]


def _parquet_rows(dataset: Path, column: str) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    return [np.asarray(pq.read_table(path)[column].to_pylist())
            for path in sorted((dataset / "data").rglob("*.parquet"))]


@pytest.mark.parametrize("mode", ("uniform_left", "uniform_double", "adaptive_left"))
def test_sidecar_schema_mapping_and_decoding(tmp_path, raw_task, mode):
    report = _run(raw_task, tmp_path / "out", SplineConfig(enabled=True, mode=mode))
    dataset, sidecar = Path(report["dataset"]), Path(report["sidecar"]["path"])
    assert sidecar.name == f"classify_blocks_30hz_cleanup_spline_{mode}"
    manifest = json.loads((sidecar / "manifest.json").read_text())
    assert manifest["source_dataset"] == str(dataset)  # not the staging path
    assert manifest["mode"] == mode and manifest["continuous_controls"] is True
    encoder = load_encoder(sidecar / "encoder.json")
    assert encoder.config.action_dim == 7 and encoder.config.frequency_hz == 30.0
    episodes, _ = load_lerobot_episodes(dataset, ActionAlignment())
    actions = _parquet_rows(dataset, "action")
    for episode, recomputed in zip(episodes, iter_encoded_episodes(encoder, episodes)):
        with np.load(sidecar / f"data/episode_{episode.episode_index:06d}.npz") as data:
            stored = {key: data[key] for key in data.files}
        np.testing.assert_array_equal(stored["frame_index"], episode.frame_indices)
        np.testing.assert_array_equal(stored["controls"],
                                      np.stack([r.control_points for r in recomputed.splines]))
        np.testing.assert_array_equal(stored["knots"], np.stack([r.knots for r in recomputed.splines]))
        assert ("duration_steps" in stored) == (mode != "uniform_double")
        assert ("control_duration_interleaved" in stored) == (mode == "adaptive_left")
        np.testing.assert_array_equal(episode.actions, actions[episode.episode_index])
    verification = report["sidecar"]["verification"]
    assert verification["records"] == sum(len(a) for a in actions)
    per_dimension = verification["fitting_error"]["max_absolute_error_per_dimension"]
    # Joint deltas jump 0 -> 0.023 rad/step at onset and around the pause.
    assert verification["all_finite"] and max(per_dimension[:6]) < 0.01
    assert per_dimension[6] < 0.5  # the synthetic gripper is a 0.5 step
    provenance = json.loads((sidecar / "preprocessing_provenance.json").read_text())
    assert provenance["dataset"] == str(dataset) and provenance["train_episodes"] == [0, 1, 2]
    assert provenance["action_transforms_in_sidecar"] == []
    assert "repeat" in provenance["tail_policy"]


def test_sidecar_never_changes_lerobot_actions(tmp_path, raw_task):
    plain = Path(_run(raw_task, tmp_path / "a", SplineConfig())["dataset"])
    with_spline = Path(_run(raw_task, tmp_path / "b", SplineConfig(
        enabled=True, normalization="q01_q99", quantize=True))["dataset"])
    for name in ("action", "observation.state"):
        for a, b in zip(_parquet_rows(plain, name), _parquet_rows(with_spline, name)):
            np.testing.assert_array_equal(a, b)


def test_normalization_and_quantization_use_training_episodes_only(tmp_path, raw_task):
    spline = SplineConfig(enabled=True, mode="uniform_left", normalization="q01_q99",
                          quantize=True, validation_episodes=1)
    report = _run(raw_task, tmp_path / "out", spline)
    dataset, sidecar = Path(report["dataset"]), Path(report["sidecar"]["path"])
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["splits"] == {"train": "0:2", "validation": "2:3"}
    actions = _parquet_rows(dataset, "action")
    train = np.concatenate(actions[:2])
    manifest = json.loads((sidecar / "manifest.json").read_text())
    np.testing.assert_allclose(manifest["alignment"]["low"], np.quantile(train, .01, axis=0))
    np.testing.assert_allclose(manifest["alignment"]["high"], np.quantile(train, .99, axis=0))
    assert not np.allclose(manifest["alignment"]["low"],
                           np.quantile(np.concatenate(actions), .01, axis=0))
    encoder = load_encoder(sidecar / "encoder.json")
    reference = create_encoder(encoder.config)
    alignment = ActionAlignment(low=tuple(manifest["alignment"]["low"]),
                                high=tuple(manifest["alignment"]["high"]))
    episodes, _ = load_lerobot_episodes(dataset, alignment)
    from data.splines.lerobot import calibrate_from_lerobot

    calibrate_from_lerobot(reference, episodes[:2])
    np.testing.assert_array_equal(encoder.quantizer.low, reference.quantizer.low)
    np.testing.assert_array_equal(encoder.quantizer.high, reference.quantizer.high)
    assert "dequantized_error" in report["sidecar"]["verification"]


def test_abs_ee_sidecar_reports_quaternion_norms_and_raw_sign_flips(tmp_path, raw_task):
    # The fixture flips the raw quaternion sign on every third sample.
    raw = _run(raw_task, tmp_path / "raw", SplineConfig(enabled=True), setups=("state-abs-ee",))
    verification = raw["sidecar"]["verification"]
    assert "renormalize" in verification["decoded_quaternion_norm"]["note"]
    assert verification["reference_quaternion_sign_flips"] > 0 and "continuous" in verification["warning"]
    continuous = _run(raw_task, tmp_path / "continuous", SplineConfig(enabled=True), setups=(),
                      custom=Setup("joint", "abs-ee", quaternion="continuous"))
    verification = continuous["sidecar"]["verification"]
    assert verification["reference_quaternion_sign_flips"] == 0 and "warning" not in verification
    assert verification["decoded_quaternion_norm"]["max_abs_deviation_from_unit"] < 0.05


def test_held_out_episodes_are_reported_separately(tmp_path, raw_task):
    report = _run(raw_task, tmp_path / "out", SplineConfig(
        enabled=True, quantize=True, validation_episodes=1))
    verification = report["sidecar"]["verification"]
    assert set(verification["fitting_error_by_split"]) == {"train", "held_out"}
    assert verification["controls_outside_calibration"]["train"] == 0


def test_incompatible_spline_settings_are_rejected_before_writing(tmp_path, raw_task):
    state = tmp_path / "wrong.json"
    save_encoder(create_encoder(UniformLeftBSplineConfig(
        action_dim=8, chunk_size=16, frequency_hz=30.0, num_basis=11, span_length_steps=2)), state)
    cases = (
        (SplineConfig(enabled=True, encoder_state=str(state)), "action_dim"),
        (SplineConfig(enabled=True, preset="piper_chunk50"), "frequency_hz"),
        (SplineConfig(enabled=True, exclude_knot_dimensions=(6,)), "adaptive"),
        (SplineConfig(enabled=True, validation_episodes=3), "training episode"),
    )
    for index, (spline, message) in enumerate(cases):
        output = tmp_path / f"case{index}"
        with pytest.raises(ValueError, match=message):
            _run(raw_task, output, spline)
        assert not any(output.rglob("*.parquet")) if output.exists() else True


def test_compatible_saved_encoder_is_reused(tmp_path, raw_task):
    state = tmp_path / "encoder.json"
    config = UniformLeftBSplineConfig(action_dim=7, chunk_size=8, frequency_hz=30.0,
                                      num_basis=7, span_length_steps=2)
    save_encoder(create_encoder(config), state)
    report = _run(raw_task, tmp_path / "out", SplineConfig(enabled=True, encoder_state=str(state)))
    assert report["sidecar"]["tokenizer_id"] == config.fingerprint
    assert load_encoder(Path(report["sidecar"]["path"]) / "encoder.json").config == config
