import subprocess
import sys

import spline_encoder
from spline_encoder import fitting


def test_declared_public_api_is_unique_and_available():
    assert len(spline_encoder.__all__) == len(set(spline_encoder.__all__))
    assert all(hasattr(spline_encoder, name) for name in spline_encoder.__all__)
    assert "make_uniform_left_clamped_knots" in fitting.__all__


def test_package_module_exposes_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "spline_encoder", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Command-line LeRobot action-to-spline encoding" in completed.stdout
