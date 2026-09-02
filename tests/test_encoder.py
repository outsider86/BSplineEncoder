import numpy as np
import pytest

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
    assert preset("piper_chunk50", "uniform_left").fingerprint == "3f7924fa97a514c5"
    assert preset("piper_chunk50", "adaptive_left").fingerprint == "88cf71f30492871b"
    assert preset("libero_chunk8", "uniform_left").fingerprint == "523cfb2a6433aad3"


@pytest.mark.parametrize(
    "mode,input_count", [("uniform_double", 8), ("uniform_left", 14)]
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


def test_uniform_left_is_exact_prefix_of_long_double_fit():
    values = episode(14)
    left = create_encoder(preset("libero_chunk8", "uniform_left"))
    double = create_encoder(UniformDoubleBSplineConfig(
        action_dim=7, chunk_size=14, frequency_hz=10, degree=3,
        num_basis=10, span_length_steps=2,
    ))
    full = double.fit_control_points(values)
    np.testing.assert_allclose(left.fit_full_control_points(values), full, atol=1e-13, rtol=0)
    np.testing.assert_allclose(left.fit_control_points(values), full[:7], atol=1e-13, rtol=0)
    np.testing.assert_allclose(left.encode_chunk(values).decode(),
                               double.basis[:8] @ full, atol=1e-13, rtol=0)


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
        encoder.encode_chunk(np.zeros((8, 7)))
    with pytest.raises(RuntimeError, match="calibration"):
        encoder.encode_chunk(np.zeros((14, 7)), quantize=True)
