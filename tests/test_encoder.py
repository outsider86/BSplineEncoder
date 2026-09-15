from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import BSpline

from spline_encoder import (
    AdaptiveLeftBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    create_encoder,
    encode_action_chunk,
    load_encoder,
    preset,
    save_encoder,
)
from spline_encoder.adaptive import fit_adaptive_bspline


def episode(length=80, dim=7):
    steps = np.linspace(0.0, 4.0, length)
    return np.stack([np.sin(steps + index / 3) for index in range(dim)], axis=1)


def windows(values, count):
    for step in range(len(values)):
        chunk = values[step : step + count]
        if len(chunk) < count:
            chunk = np.concatenate((chunk, np.repeat(chunk[-1:], count - len(chunk), 0)))
        yield chunk


def test_validated_preset_fingerprints_are_preserved():
    assert preset("piper_chunk50", "uniform_double").fingerprint == "b2211e5e8b95d867"
    assert preset("piper_chunk50", "uniform_left").fingerprint == "4c16dc0255530685"
    assert preset("piper_chunk50", "adaptive_left").fingerprint == "88cf71f30492871b"
    assert preset("libero_chunk8", "uniform_left").fingerprint == "0e47f7774b762748"

    legacy = replace(
        preset("piper_chunk50", "uniform_left"),
        implementation_version="uniform_left_extended_double_fit_v1",
    )
    assert legacy.fingerprint == "3f7924fa97a514c5"


@pytest.mark.parametrize(
    "mode,input_count", [("uniform_double", 8), ("uniform_left", 8)]
)
def test_continuous_first_chunk_encoding_and_optional_quantization(mode, input_count):
    values = episode()
    encoder = create_encoder(preset("libero_chunk8", mode))
    encoder.calibrate(windows(values, input_count))
    result = encoder.encode_chunk(values[:input_count])
    assert result.control_points.shape == (7, 7)
    assert result.tokens is None
    assert result.decode().shape == (8, 7)

    quantized = encoder.encode_chunk(values[:input_count], quantize=True)
    assert quantized.tokens.shape == (7, 7)
    assert quantized.dequantized_control_points.shape == (7, 7)
    assert quantized.decode(dequantized=True).shape == (8, 7)


def test_uniform_left_fits_its_left_clamped_basis_directly():
    values = episode(8)
    left = create_encoder(preset("libero_chunk8", "uniform_left"))
    expected_knots = np.asarray([0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14], dtype=float)
    np.testing.assert_array_equal(left.knots, expected_knots)
    assert left.config.input_chunk_size == left.config.chunk_size == 8
    assert not hasattr(left, "full_knots")
    assert not hasattr(left, "retained_control_indices")

    fit_steps = np.arange(9, dtype=float)
    design = BSpline.design_matrix(
        fit_steps, expected_knots, left.config.degree, extrapolate=False
    ).toarray()
    padded_values = np.concatenate((values, values[-1:]), axis=0)
    expected = np.linalg.solve(
        design.T @ design
        + left.config.regularization * np.eye(left.config.num_basis),
        design.T @ padded_values,
    )
    np.testing.assert_allclose(left.fit_control_points(values), expected, atol=1e-13, rtol=0)


def test_explicit_legacy_uniform_left_preserves_extended_double_fit(tmp_path):
    values = episode(14)
    config = replace(
        preset("libero_chunk8", "uniform_left"),
        implementation_version="uniform_left_extended_double_fit_v1",
    )
    left = create_encoder(config)
    double = create_encoder(UniformDoubleBSplineConfig(
        action_dim=7, chunk_size=14, frequency_hz=10, degree=3,
        num_basis=10, span_length_steps=2,
    ))
    full = double.fit_control_points(values)
    assert config.input_chunk_size == 14
    np.testing.assert_allclose(left.fit_full_control_points(values), full, atol=1e-13, rtol=0)
    np.testing.assert_allclose(left.fit_control_points(values), full[:7], atol=1e-13, rtol=0)
    restored = load_encoder(save_encoder(left, tmp_path / "legacy-left.json"))
    assert restored.config.implementation_version == config.implementation_version
    np.testing.assert_allclose(restored.fit_control_points(values), full[:7], atol=1e-13, rtol=0)


def test_different_degrees_and_dimensions_are_supported():
    config = UniformDoubleBSplineConfig(
        action_dim=3,
        chunk_size=12,
        frequency_hz=30,
        degree=2,
        num_basis=6,
        span_length_steps=3,
    )
    result = encode_action_chunk(episode(12, 3), config)
    assert result.control_points.shape == (6, 3)
    assert result.knots.shape == (9,)
    assert result.decode().shape == (12, 3)


def test_adaptive_chunk_and_episode_are_observation_aligned():
    values = episode(64)
    encoder = create_encoder(preset("libero_chunk8", "adaptive_left"))
    records = encoder.encode_episode(values)
    assert len(records) == len(values)
    assert [record.observation_index for record in records] == list(range(len(values)))
    assert records[0].control_points.shape == (7, 7)
    assert records[0].duration_steps.shape == (7,)
    assert records[0].decode().shape[1] == 7
    standalone = encoder.encode_chunk(values[:8])
    assert standalone.observation_index == 0


def test_adaptive_knot_insertion_can_exclude_a_dimension():
    steps = np.linspace(0.0, 4.0, 120)
    arm = np.stack((np.sin(steps), np.cos(steps), 0.2 * steps), axis=1)
    gripper = np.where((np.arange(len(steps)) // 8) % 2, 1.0, 0.0)[:, None]
    actions = np.concatenate((arm, gripper), axis=1)
    common = dict(degree=3, max_error=0.01, smoothing=1e-12)

    arm_fit = fit_adaptive_bspline(arm, **common)
    excluded_fit = fit_adaptive_bspline(
        actions,
        knot_insertion_excluded_dimensions=(3,),
        **common,
    )
    all_dimension_fit = fit_adaptive_bspline(actions, **common)

    # Knot insertion sees exactly the arm fit, while the final spline retains
    # every original dimension in its original column position.
    np.testing.assert_array_equal(excluded_fit.knots, arm_fit.knots)
    np.testing.assert_allclose(
        excluded_fit.control_points[:, :3], arm_fit.control_points, atol=1e-13, rtol=0
    )
    assert excluded_fit.control_points.shape[1] == actions.shape[1]
    assert len(all_dimension_fit.knots) > len(excluded_fit.knots)
    assert np.isfinite(excluded_fit.spline(steps)).all()


def test_adaptive_excluded_dimensions_are_validated_and_serialized(tmp_path):
    config = AdaptiveLeftBSplineConfig(
        action_dim=7,
        chunk_size=8,
        frequency_hz=10,
        degree=3,
        num_basis=7,
        span_length_steps=2,
        knot_insertion_excluded_dimensions=(6,),
    )
    assert config.fingerprint != preset("libero_chunk8", "adaptive_left").fingerprint
    encoder = create_encoder(config)
    restored = load_encoder(save_encoder(encoder, tmp_path / "encoder.json"))
    assert restored.config.knot_insertion_excluded_dimensions == (6,)

    with pytest.raises(ValueError, match="unique"):
        AdaptiveLeftBSplineConfig(
            action_dim=7, chunk_size=8, num_basis=7, span_length_steps=2,
            knot_insertion_excluded_dimensions=(6, 6),
        )
    with pytest.raises(ValueError, match="valid action dimensions"):
        AdaptiveLeftBSplineConfig(
            action_dim=7, chunk_size=8, num_basis=7, span_length_steps=2,
            knot_insertion_excluded_dimensions=(7,),
        )
    with pytest.raises(ValueError, match="at least one"):
        AdaptiveLeftBSplineConfig(
            action_dim=2, chunk_size=8, num_basis=7, span_length_steps=2,
            knot_insertion_excluded_dimensions=(0, 1),
        )


def test_adaptive_control_duration_rows_round_trip_to_knots_and_controls():
    values = episode(64)
    encoder = create_encoder(AdaptiveLeftBSplineConfig(
        action_dim=7,
        chunk_size=8,
        frequency_hz=10,
        degree=3,
        num_basis=7,
        span_length_steps=2,
        knot_insertion_excluded_dimensions=(6,),
    ))
    record = encoder.encode_episode(values)[0]
    rows = record.interleaved_control_durations()
    assert rows.shape == (7, 8)
    np.testing.assert_array_equal(rows[:, :7], record.control_points)
    np.testing.assert_array_equal(rows[:, 7], record.duration_steps)
    geometry = encoder.geometry(rows[:, 7].astype(np.int64))
    np.testing.assert_array_equal(
        geometry.knot_steps,
        np.rint(record.knot_steps).astype(np.int64),
    )
    reconstructed = record.__class__(
        control_points=rows[:, :7],
        knots=geometry.knot_steps * record.sample_period,
        degree=record.degree,
        sample_period=record.sample_period,
        executable_steps=record.executable_steps,
        tokenizer_id=record.tokenizer_id,
        duration_steps=rows[:, 7].astype(np.int64),
        metadata=record.metadata,
    )
    np.testing.assert_allclose(reconstructed.decode(), record.decode(), atol=1e-13, rtol=0)


def test_state_roundtrip_and_legacy_schema(tmp_path):
    encoder = create_encoder(preset("libero_chunk8", "uniform_double"))
    values = episode()
    encoder.calibrate(windows(values, 8))
    path = save_encoder(encoder, tmp_path / "encoder.json")
    restored = load_encoder(path)
    np.testing.assert_array_equal(
        restored.encode_chunk(values[:8], quantize=True).tokens,
        encoder.encode_chunk(values[:8], quantize=True).tokens,
    )

    # SplineEncoder directly accepts original action_tokenization schema-v1.
    import json
    payload = json.loads(path.read_text())
    payload["library"] = "action_tokenization"
    payload.pop("quantization_available")
    legacy = tmp_path / "action_tokenizer.json"
    legacy.write_text(json.dumps(payload))
    assert load_encoder(legacy).tokenizer_id == encoder.tokenizer_id


def test_invalid_shapes_fail_early():
    encoder = create_encoder(preset("libero_chunk8", "uniform_left"))
    with pytest.raises(ValueError, match="shape"):
        encoder.encode_chunk(np.zeros((14, 7)))
    with pytest.raises(RuntimeError, match="calibration"):
        encoder.encode_chunk(np.zeros((8, 7)), quantize=True)
