"""Regression tests for the 2026-10 implementation review (docs/implementation_review.md)."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg

from spline_encoder import (
    AdaptiveLeftBSplineConfig,
    UniformDoubleBSplineConfig,
    UniformLeftBSplineConfig,
    create_encoder,
    preset,
    subdivide_parameters,
    verify_spline_parameters,
)
from spline_encoder.lerobot import LeRobotEpisode, iter_encoded_episodes

REPOSITORY = Path(__file__).resolve().parents[1]


def smooth(length, dim=7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0, length)
    return np.stack([np.sin((i + 1) * t + rng.uniform(0, 3)) for i in range(dim)], axis=1)


def augmented_lstsq(basis, regularization, values):
    n = basis.shape[1]
    design = np.vstack((basis, np.sqrt(regularization) * np.eye(n)))
    target = np.vstack((values, np.zeros((n, values.shape[1]))))
    return scipy.linalg.lstsq(design, target, lapack_driver="gelsd")[0]


# --- Cached uniform solvers -------------------------------------------------

@pytest.mark.parametrize("name", ("libero", "libero_chunk8", "robotwin", "piper_chunk50"))
@pytest.mark.parametrize("version", ("double", "direct", "legacy"))
def test_cached_uniform_solvers_match_stable_augmented_least_squares(name, version):
    mode = "uniform_double" if version == "double" else "uniform_left"
    config = preset(name, mode)
    if version == "legacy":
        config = replace(config, implementation_version="uniform_left_extended_double_fit_v1")
    encoder = create_encoder(config)
    values = np.random.default_rng(7).normal(size=(encoder._fit_basis.shape[0], config.action_dim))
    reference = augmented_lstsq(encoder._fit_basis, config.regularization, values)
    np.testing.assert_allclose(encoder._solver @ values, reference, atol=1e-12, rtol=0)


@pytest.mark.parametrize("cls,span", (
    (UniformDoubleBSplineConfig, None),
    (UniformDoubleBSplineConfig, 1),
    (UniformLeftBSplineConfig, 1),
))
def test_unregularized_rank_deficient_uniform_layout_is_rejected(cls, span):
    # Before the review the cached inverse silently returned controls ~1e18.
    chunk = 4
    num_basis = 8 if span is None else chunk + 3
    config = cls(action_dim=1, chunk_size=chunk, frequency_hz=10.0, degree=3,
                 num_basis=num_basis, span_length_steps=span, regularization=0.0)
    with pytest.raises(ValueError, match="rank deficient"):
        create_encoder(config)
    regularized = create_encoder(replace(config, regularization=1e-4))
    controls = regularized.fit_control_points(np.arange(chunk, dtype=float)[:, None])
    assert np.all(np.isfinite(controls)) and np.abs(controls).max() < 10


# --- Direct versus legacy uniform-left ---------------------------------------

def test_direct_and_legacy_uniform_left_are_different_procedures():
    direct_config = preset("piper_chunk50", "uniform_left")
    legacy_config = replace(direct_config, implementation_version="uniform_left_extended_double_fit_v1")
    direct, legacy = create_encoder(direct_config), create_encoder(legacy_config)
    assert (direct_config.input_chunk_size, legacy_config.input_chunk_size) == (50, 56)
    np.testing.assert_array_equal(direct.knots, legacy.knots)

    window = smooth(56, seed=1)
    future_changed = window.copy()
    future_changed[50:] += 0.5
    # The legacy fit consumes six future actions; the direct fit cannot see them.
    assert np.abs(legacy.fit_control_points(window)
                  - legacy.fit_control_points(future_changed)).max() > 1e-3
    assert np.abs(direct.fit_control_points(window[:50])
                  - legacy.fit_control_points(window)).max() > 1e-3

    # Legacy retained controls reproduce its extended double-clamped fit
    # exactly on the 50-step executable domain.
    full = legacy.fit_full_control_points(window)
    from scipy.interpolate import BSpline
    extended = BSpline(legacy.full_knots * direct_config.delta_t, full, 3, axis=0)
    retained = legacy.encode_chunk(window).scipy_spline()
    x = np.linspace(0.0, 1.0, 301)
    np.testing.assert_allclose(retained(x), extended(x), atol=1e-13, rtol=0)


# --- subdivide_parameters (hypothesis 1) --------------------------------------

@pytest.mark.parametrize("span", (None, 2))
def test_subdivide_parameters_children_stay_aligned_with_native_samples(span):
    chunk = 17 if span is None else 16
    config = UniformDoubleBSplineConfig(action_dim=2, chunk_size=chunk, frequency_hz=10.0,
                                        num_basis=11, span_length_steps=span)
    result = create_encoder(config).encode_chunk(smooth(chunk, dim=2))
    last_valid = chunk - 2 if span is None else chunk - 1
    for step in range(1, last_valid + 1):
        left, right = subdivide_parameters(result, step)
        np.testing.assert_allclose(np.concatenate((left.decode(), right.decode())),
                                   result.decode(), atol=1e-14, rtol=0)
        assert verify_spline_parameters(left)["passed"]
        assert verify_spline_parameters(right)["passed"]
    if span is None:
        # The final inclusive sample sits on the domain end: clear error.
        with pytest.raises(ValueError, match="strictly inside the executable horizon"):
            subdivide_parameters(result, chunk - 1)


def test_subdivided_uniform_left_left_child_is_labelled_double_clamped():
    result = create_encoder(preset("libero_chunk8", "uniform_left")).encode_chunk(smooth(8))
    left, right = subdivide_parameters(result, 3)
    assert np.count_nonzero(left.knots == left.knots[-1]) == left.degree + 1
    assert left.metadata["boundary_condition"] == "double_clamped"
    assert right.metadata["boundary_condition"] == "left_clamped_right_open"


# --- Verification of malformed results -----------------------------------------

def test_verification_reports_rather_than_crashes_on_malformed_results():
    encoder = create_encoder(preset("libero_chunk8", "adaptive_left"))
    record = encoder.encode_episode(smooth(40))[5]
    short = replace(record, duration_steps=record.duration_steps[:-1])
    report = verify_spline_parameters(short)
    assert not report["passed"] and not report["checks"]["duration_knot_round_trip"]

    domain_steps = int(round(record.knot_steps[record.num_basis]))
    beyond = replace(record, executable_steps=domain_steps + 1)
    report = verify_spline_parameters(beyond)
    assert report["checks"]["executable_samples_inside_domain"]
    assert not report["checks"]["executable_horizon_inside_domain"]
    assert not report["passed"]
    # The flagged result really does decode outside its domain when resampled.
    assert np.isnan(beyond.decode(samples=200)).any()


@pytest.mark.parametrize("mode", ("uniform_double", "uniform_left", "adaptive_left"))
def test_encoder_outputs_satisfy_the_horizon_check_and_unit_conventions(mode):
    config = preset("piper_chunk50", mode)
    encoder = create_encoder(config)
    actions = smooth(80 if mode == "adaptive_left" else 50)
    result = (encoder.encode_episode(actions)[9] if mode == "adaptive_left"
              else encoder.encode_chunk(actions))
    assert verify_spline_parameters(result)["passed"]
    # Seconds versus steps: knots are seconds, knot_steps are native steps.
    np.testing.assert_allclose(result.knots, result.knot_steps * config.delta_t, atol=1e-15)
    if result.duration_steps is not None:
        distinct = np.r_[0, np.cumsum(result.duration_steps)]
        np.testing.assert_allclose(np.unique(result.knot_steps), distinct, atol=1e-9)
        assert result.executable_steps == distinct[config.num_basis - config.degree]


# --- Short episodes and dataset alignment ---------------------------------------

def test_short_adaptive_episodes_fail_loudly_in_dataset_encoding():
    config = AdaptiveLeftBSplineConfig(action_dim=2, chunk_size=8, frequency_hz=10.0,
                                       degree=3, num_basis=7, span_length_steps=2)
    encoder = create_encoder(config)
    assert encoder.encode_episode(smooth(3, dim=2)) == []
    assert len(encoder.encode_episode(smooth(4, dim=2))) == 4
    episode = LeRobotEpisode(0, np.arange(3), smooth(3, dim=2))
    with pytest.raises(RuntimeError, match="produced 0 records for 3 observations"):
        list(iter_encoded_episodes(encoder, [episode]))
    # Without tail padding late observations lack future knot spans.
    unpadded = create_encoder(replace(config, end_padding=False))
    assert len(unpadded.encode_episode(smooth(60, dim=2))) < 60


def test_uniform_short_episode_windows_repeat_the_last_action():
    encoder = create_encoder(preset("libero_chunk8", "uniform_double"))
    actions = smooth(3)
    episode = LeRobotEpisode(0, np.arange(3), actions)
    splines = next(iter_encoded_episodes(encoder, [episode])).splines
    assert [item.observation_index for item in splines] == [0, 1, 2]
    padded = np.concatenate((actions[2:], np.repeat(actions[-1:], 7, axis=0)))
    np.testing.assert_array_equal(splines[2].control_points,
                                  encoder.encode_chunk(padded).control_points)


@pytest.mark.parametrize("example", ("encode_chunk.py", "fit_control_points.py"))
def test_offline_examples_run(example):
    subprocess.run([sys.executable, str(REPOSITORY / "examples" / example)],
                   check=True, capture_output=True, text=True, cwd=REPOSITORY)


def test_visual_verification_command_passes_its_numerical_checks(tmp_path):
    pytest.importorskip("matplotlib")
    import json

    completed = subprocess.run(
        [sys.executable, str(REPOSITORY / "scripts" / "visual_verification.py"),
         "--output-dir", str(tmp_path)],
        check=False, capture_output=True, text=True, cwd=REPOSITORY,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "visual_verification_metrics.json").read_text())
    assert report["passed"] and all(report["checks"].values())
    for name in report["figures"]:
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert report["metrics"]["transfer_requantization_clipped_controls"] == 0
