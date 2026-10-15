"""Exact equivalence with the original preprocessing scripts (``scripts/legacy_piper``).

The legacy scripts are executed unmodified in a subprocess (they import each
other as top-level modules), so the package never needs ``sys.path`` edits.
Skipped when ``scripts/legacy_piper`` is not present.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("cv2")

from piper_fixtures import make_raw_episode  # noqa: E402
from data import ACTIONS, PRESETS, STATES, RunConfig, Setup  # noqa: E402
from data.config import default_camera  # noqa: E402
from data.conversion.episode import clip_episode, load_episode  # noqa: E402

LEGACY = Path(__file__).resolve().parents[1] / "scripts" / "legacy_piper"
pytestmark = pytest.mark.skipif(not (LEGACY / "preprocess_piper.py").is_file(),
                                reason="legacy scripts not present")

SCRIPT = r"""
import json, sys
from pathlib import Path
import numpy as np
import preprocess_piper as legacy
from convert_piper_to_lerobot_delta_ee import _load_episode as delta_ee_load
from convert_piper_to_lerobot import _load_episode as joint_load

episode, combos, output = Path(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
arrays = {}
for index, (setup, run) in enumerate(combos):
    setup, run = legacy.Setup(**setup), legacy.RunConfig(**run)
    aligned = legacy.load_episode(episode, setup, run)
    cropped, metric = legacy.clip_episode(aligned, setup, run, 0)
    arrays[f"{index}/state"] = cropped.state
    arrays[f"{index}/action"] = cropped.action
    arrays[f"{index}/cutoff"] = np.asarray(metric["removed_frames"])
    for name, values in cropped.camera_indices.items():
        arrays[f"{index}/camera/{name}"] = values
arrays["delta_ee/state"] = delta_ee_load(episode, 30.0).state
arrays["delta_ee/action"] = delta_ee_load(episode, 30.0).action
base = joint_load(episode, 30.0, "action", "state_last_command", 0)
arrays["joint/state"], arrays["joint/action"] = base.state, base.action
np.savez(output, **arrays)
"""


def _combinations():
    combos = [Setup(state, action, offset, quaternion, clock)
              for state in STATES for action in ACTIONS
              for offset in ((1,) if action.startswith("delta-") else (0, 1))
              for quaternion in ("raw", "continuous") for clock in ("raw", "target")]
    runs = [RunConfig(), RunConfig(cleanup_motion="command-joint", preroll_seconds=.1),
            RunConfig(cleanup=False), RunConfig(cleanup_motion="measured-ee", min_remaining_seconds=1.)]
    pairs = [(setup, runs[i % len(runs)]) for i, setup in enumerate(combos)]
    pairs += [(setup, RunConfig()) for setup, _ in PRESETS.values()]
    return pairs


def test_package_matches_legacy_scripts_exactly(tmp_path):
    episode = make_raw_episode(tmp_path / "raw" / "episode")
    pairs = _combinations()
    payload = [({k: v for k, v in asdict(setup).items()
                 if k in ("state", "action", "action_offset", "quaternion", "last_command_clock")},
                asdict(run)) for setup, run in pairs]
    output = tmp_path / "legacy.npz"
    # No bytecode: the legacy source directory is left exactly as provided.
    environment = {**os.environ, "PYTHONPATH": str(LEGACY), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run([sys.executable, "-c", SCRIPT, str(episode), json.dumps(payload), str(output)],
                   check=True, cwd=tmp_path, env=environment, capture_output=True, text=True)
    cameras = (default_camera("global"), default_camera("hand"))
    with np.load(output) as legacy:
        for index, (setup, run) in enumerate(pairs):
            aligned = load_episode(episode, setup, run, cameras)
            cropped, metric = clip_episode(aligned, setup, run, 0)
            np.testing.assert_array_equal(cropped.state, legacy[f"{index}/state"])
            np.testing.assert_array_equal(cropped.action, legacy[f"{index}/action"])
            assert metric["removed_frames"] == int(legacy[f"{index}/cutoff"])
            for name in ("global", "hand"):
                np.testing.assert_array_equal(cropped.camera_indices[name],
                                              legacy[f"{index}/camera/{name}"])
        no_cleanup = RunConfig(cleanup=False)
        delta = load_episode(episode, PRESETS["ee-delta-ee"][0], no_cleanup, cameras).episode
        np.testing.assert_array_equal(delta.state, legacy["delta_ee/state"])
        np.testing.assert_array_equal(delta.action, legacy["delta_ee/action"])
        joint = load_episode(episode, Setup("joint-last-command", "command", 0), no_cleanup,
                             cameras).episode
        np.testing.assert_array_equal(joint.state, legacy["joint/state"])
        np.testing.assert_array_equal(joint.action, legacy["joint/action"])
