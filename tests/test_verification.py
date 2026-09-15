import json
from pathlib import Path
import subprocess
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from spline_encoder import (
    create_encoder,
    plot_reconstruction,
    preset,
    reconstruction_metrics,
    save_reconstruction_plot,
    verify_reconstruction,
    verify_spline_parameters,
)


def smooth_actions(length, dim=7):
    steps = np.linspace(0.0, 2.0, length)
    return np.stack([
        np.sin((index + 1) * steps / 3.0) + 0.1 * index * steps
        for index in range(dim)
    ], axis=1)


def test_reconstruction_metrics_are_exact_and_json_serializable():
    reference = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    reconstructed = np.asarray([[1.0, 1.0], [0.0, 5.0]])
    metrics = reconstruction_metrics(reference, reconstructed)
    assert metrics.sample_count == 2
    assert metrics.action_dim == 2
    assert metrics.mae == pytest.approx(1.25)
    assert metrics.rmse == pytest.approx(1.5)
    assert metrics.max_absolute_error == 2.0
    assert json.loads(json.dumps(metrics.to_dict()))["mae_per_action_dimension"] == [1.5, 1.0]


@pytest.mark.parametrize("mode,input_steps", (("uniform_double", 8), ("uniform_left", 8)))
def test_result_invariants_and_thresholded_reconstruction(mode, input_steps):
    actions = smooth_actions(input_steps)
    result = create_encoder(preset("libero_chunk8", mode)).encode_chunk(actions)
    invariants = verify_spline_parameters(result)
    assert invariants["passed"]
    assert all(invariants["checks"].values())
    report = verify_reconstruction(
        result,
        actions[: result.executable_steps],
        max_absolute_error=1.0,
    )
    assert report["passed"]
    assert report["error_within_tolerance"]
    failed = verify_reconstruction(
        result,
        actions[: result.executable_steps],
        max_absolute_error=0.0,
    )
    assert not failed["passed"]


def test_adaptive_duration_geometry_passes_independent_round_trip():
    actions = smooth_actions(64)
    result = create_encoder(preset("libero_chunk8", "adaptive_left")).encode_episode(actions)[17]
    report = verify_spline_parameters(result)
    assert report["passed"]
    assert report["checks"]["duration_knot_round_trip"]


def test_verification_rejects_ambiguous_or_nonfinite_inputs():
    result = create_encoder(preset("libero_chunk8", "uniform_double")).encode_chunk(
        smooth_actions(8)
    )
    with pytest.raises(ValueError, match="reference must have shape"):
        verify_reconstruction(result, smooth_actions(7))
    with pytest.raises(ValueError, match="finite"):
        reconstruction_metrics([[np.nan]], [[0.0]])


def test_plot_contains_expected_layers_and_saves_nonempty_png(tmp_path):
    actions = smooth_actions(8)
    result = create_encoder(preset("libero_chunk8", "uniform_left")).encode_chunk(actions)
    reference = actions[: result.executable_steps]
    figure, axes = plot_reconstruction(
        result,
        reference,
        dimensions=(0, 3),
        dense_samples=101,
    )
    try:
        assert len(axes) == 2
        for dimension, axis in zip((0, 3), axes):
            labels = {line.get_label() for line in axis.lines}
            assert {"continuous spline", "target actions", "knots"} <= labels
            target_line = next(line for line in axis.lines if line.get_label() == "target actions")
            np.testing.assert_allclose(target_line.get_ydata(), reference[:, dimension])
            assert len(next(
                collection for collection in axis.collections
                if collection.get_label() == "control coefficients"
            ).get_offsets()) > 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)

    path = save_reconstruction_plot(
        result,
        tmp_path / "diagnostics" / "chunk.png",
        reference,
        dimensions=(0, 3),
    )
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.stat().st_size > 10_000


def test_verify_chunk_cli_writes_machine_readable_report_and_plot(tmp_path):
    actions_path = tmp_path / "actions.npy"
    report_path = tmp_path / "report.json"
    plot_path = tmp_path / "plot.png"
    np.save(actions_path, smooth_actions(8))
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "verify_chunk.py"),
            str(actions_path),
            "--preset",
            "libero_chunk8",
            "--mode",
            "uniform_left",
            "--max-error",
            "1.0",
            "--report",
            str(report_path),
            "--plot",
            str(plot_path),
            "--dimensions",
            "0,3",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    stored = json.loads(report_path.read_text())
    assert stdout == stored
    assert stored["passed"]
    assert stored["invariants"]["passed"]
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
