"""Semantic and integration tests for all preprocessing option combinations."""

from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess_piper as pipeline
from validate_piper_preprocessing import expected_from_raw, validate_dataset


@pytest.fixture(scope="module")
def raw_episode(tmp_path_factory):
    path = tmp_path_factory.mktemp("piper_async") / "episode"
    path.mkdir()
    time = np.arange(401, dtype=np.float64) / 100.0
    motion = np.maximum(0, time - 1.1)
    rotations = Rotation.from_euler("xyz", np.column_stack([motion * .12, motion * .1, .8 + motion * .25]))
    quaternions = rotations.as_quat()
    quaternions[::3] *= -1  # Same orientation, intentionally discontinuous signs.
    observations, commands = [], []
    for index, t in enumerate(time):
        observations.append({
            "arm_joint_position": np.array([motion[index] * .7] * 6),
            "arm_pos": np.array([.2 + motion[index] * .1, .3 - motion[index] * .05, .4]),
            "arm_quat": quaternions[index],
            "gripper_pos": np.array([.2 if t < 1.25 else .7]),
        })
        commands.append({
            "arm_joint_position": np.array([4 + max(0, t - .6)] * 6),
            "gripper_pos": np.array([.1 if t < .8 else .9]),
            # Deliberately unrelated command/FK EE values: measured actions
            # must never select these fields.
            "arm_pos": np.array([99., 99., 99.]), "arm_quat": np.array([0., 0., 0., 1.]),
        })
    camera_times = {
        "global": np.arange(119, dtype=np.float64) / 30.0 + .025,
        "hand": np.arange(115, dtype=np.float64) / 29.0 + .055,
    }
    for camera, timestamps in camera_times.items():
        writer = cv2.VideoWriter(str(path / f"{camera}_image.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30., (64, 48))
        assert writer.isOpened()
        for index in range(len(timestamps)):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:] = [index, 40 if camera == "global" else 150, 100]
            writer.write(frame)
        writer.release()
    payload = {
        "timestamps": time, "observations": observations, "actions": commands,
        "camera_timestamps": camera_times,
        "camera_frame_source_timestamps": camera_times,
    }
    with (path / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    return path


COMBINATIONS = [
    pipeline.Setup(state, action, offset, quaternion, clock)
    for state in pipeline.STATES
    for action in pipeline.ACTIONS
    for offset in ((1,) if action.startswith("delta-") else (0, 1))
    for quaternion in ("raw", "continuous")
    for clock in ("raw", "target")
]


@pytest.mark.parametrize("setup", COMBINATIONS)
def test_all_state_action_combinations_against_raw(raw_episode, setup):
    run = pipeline.RunConfig()
    aligned = pipeline.load_episode(raw_episode, setup, run)
    episode, metric = pipeline.clip_episode(aligned, setup, run, 0)
    state, action, cutoff, count, camera_indices = expected_from_raw(raw_episode, asdict(setup), asdict(run))
    np.testing.assert_array_equal(episode.state, state)
    np.testing.assert_allclose(episode.action, action, atol=2e-7, rtol=1e-6)
    assert metric["removed_frames"] == cutoff > 0
    assert episode.length == count - setup.action_offset - cutoff
    for camera in pipeline.CAMERAS:
        np.testing.assert_array_equal(episode.camera_indices[camera], camera_indices[camera])


@pytest.mark.parametrize("source", ("auto", "measured-joint", "measured-delta-joint", "command-joint", "measured-ee"))
@pytest.mark.parametrize("cleanup", (True, False))
def test_cleanup_override_and_disabled(raw_episode, source, cleanup):
    setup = pipeline.Setup("last-command", "abs-ee")
    run = pipeline.RunConfig(cleanup=cleanup, cleanup_motion=source, preroll_seconds=.1)
    aligned = pipeline.load_episode(raw_episode, setup, run)
    episode, metric = pipeline.clip_episode(aligned, setup, run, 0)
    state, action, cutoff, _, _ = expected_from_raw(raw_episode, asdict(setup), asdict(run))
    assert metric["removed_frames"] == cutoff
    np.testing.assert_array_equal(episode.state, state)
    np.testing.assert_array_equal(episode.action, action)
    if not cleanup:
        assert cutoff == 0


def test_command_clocks_and_action_sources_are_distinct(raw_episode):
    run = pipeline.RunConfig(cleanup=False)
    raw = pipeline.load_episode(raw_episode, pipeline.Setup("last-command", "joint"), run)
    target = pipeline.load_episode(raw_episode, pipeline.Setup("last-command", "joint", last_command_clock="target"), run)
    assert np.any(raw.episode.state != target.episode.state)
    assert np.max(np.abs(raw.episode.state - raw.episode.action)) > 1


def test_delta_reconstructs_absolute_next_pose_and_gripper(raw_episode):
    aligned = pipeline.load_episode(raw_episode, pipeline.Setup("ee", "delta-ee"), pipeline.RunConfig(cleanup=False))
    current, action = aligned.episode.state, aligned.episode.action
    current_rotation = Rotation.from_quat(current[:, 3:7])
    position = current[:, :3] + current_rotation.apply(action[:, :3])
    next_rotation = current_rotation * Rotation.from_rotvec(action[:, 3:6])
    np.testing.assert_allclose(position, aligned.measured_ee[1:, :3], atol=1e-7)
    expected_rotation = Rotation.from_quat(aligned.measured_ee[1:, 3:7])
    np.testing.assert_allclose((next_rotation.inv() * expected_rotation).magnitude(), 0, atol=1e-7)
    np.testing.assert_array_equal(action[:, 6], aligned.measured_ee[1:, 7])


def test_delta_joint_reconstructs_measured_next_joints_and_gripper(raw_episode):
    setup = pipeline.PRESETS["state-delta-joint"][0]
    aligned = pipeline.load_episode(raw_episode, setup, pipeline.RunConfig(cleanup=False))
    state, action = aligned.episode.state, aligned.episode.action
    np.testing.assert_allclose(state[:, :6] + action[:, :6], aligned.measured_joint[1:, :6], atol=1e-7)
    np.testing.assert_array_equal(action[:, 6], aligned.measured_joint[1:, 6])
    assert np.any(action[:, 6] != aligned.measured_joint[1:, 6] - state[:, 6])
    assert len(state) == len(aligned.measured_joint) - 1


@pytest.mark.parametrize("source", ("observation_delta_joint", "observation_delta_ee"))
def test_delta_cleanup_keeps_current_frame_that_initiates_motion(source):
    state = np.zeros((90, 7), np.float32)
    state[:, 6] = .3
    action = np.zeros((90, 7), np.float32)
    action[:, 6] = .3
    action[40:, 0] = .02
    config = pipeline.RunConfig().detector()
    signals = pipeline._motion_signals(state, action, config, "state", source)
    assert pipeline._detect_cutoff(signals, len(state), config) == 30  # minimum 60 remaining
    config = pipeline.RunConfig(min_remaining_seconds=1.).detector()
    signals = pipeline._motion_signals(state, action, config, "state", source)
    assert pipeline._detect_cutoff(signals, len(state), config) == 40


@pytest.mark.parametrize("preset", ("state-delta-joint", "state-delta-ee"))
def test_new_delta_preset_full_pipeline(tmp_path, raw_episode, monkeypatch, preset):
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", pipeline.resolve_ffmpeg(None))
    setup, family = pipeline.PRESETS[preset]
    output = tmp_path / family
    pipeline.convert(raw_episode.parent, output, "Test task.", setup, pipeline.RunConfig(), preset)
    report = validate_dataset(output)
    assert report["passed"] and report["videos_checked"] == 2
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["action"]["shape"] == [7]
    assert info["action_timestep_offset"] == 1


def test_no_commands_required_for_measured_only(tmp_path, raw_episode):
    path = tmp_path / "measured_only"
    path.mkdir()
    with (raw_episode / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    del payload["actions"]
    with (path / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    for camera in pipeline.CAMERAS:
        (path / f"{camera}_image.mp4").symlink_to(raw_episode / f"{camera}_image.mp4")
    aligned = pipeline.load_episode(path, pipeline.Setup("joint", "abs-ee"), pipeline.RunConfig())
    assert aligned.command is None
    with pytest.raises(ValueError, match="commands"):
        pipeline.load_episode(path, pipeline.Setup("last-command", "abs-ee"), pipeline.RunConfig())


@pytest.mark.parametrize("setup", [value[0] for value in pipeline.PRESETS.values()])
def test_named_presets(raw_episode, setup):
    setup.check()
    aligned = pipeline.load_episode(raw_episode, setup, pipeline.RunConfig())
    episode, _ = pipeline.clip_episode(aligned, setup, pipeline.RunConfig(), 0)
    state, action, _, _, _ = expected_from_raw(raw_episode, asdict(setup), asdict(pipeline.RunConfig()))
    np.testing.assert_array_equal(episode.state, state)
    np.testing.assert_allclose(episode.action, action, atol=2e-7, rtol=1e-6)


def test_legacy_delta_converter_compatibility(raw_episode):
    from convert_piper_to_lerobot_delta_ee import _load_episode

    setup = pipeline.PRESETS["ee-delta-ee"][0]
    legacy = _load_episode(raw_episode, 30.)
    unified = pipeline.load_episode(raw_episode, setup, pipeline.RunConfig(cleanup=False)).episode
    np.testing.assert_array_equal(legacy.state, unified.state)
    np.testing.assert_array_equal(legacy.action, unified.action)


def test_legacy_joint_converter_compatibility(raw_episode):
    from convert_piper_to_lerobot import _load_episode

    for name, (setup, _) in pipeline.PRESETS.items():
        if setup.action.startswith("delta-"):
            continue
        legacy = _load_episode(raw_episode, 30., pipeline.ACTION_SOURCE[setup.action],
                               pipeline.STATE_SOURCE[setup.state], setup.action_offset)
        unified = pipeline.load_episode(raw_episode, setup, pipeline.RunConfig(cleanup=False)).episode
        np.testing.assert_array_equal(legacy.state, unified.state)
        np.testing.assert_array_equal(legacy.action, unified.action)


@pytest.mark.parametrize("action", ("delta-joint", "delta-ee"))
def test_invalid_delta_offset_and_thresholds(action):
    with pytest.raises(ValueError, match="requires"):
        pipeline.Setup("joint", action, 0).check()
    with pytest.raises(ValueError, match="positive"):
        pipeline.RunConfig(fps=float("nan")).detector()


def test_idle_cutoff_is_per_episode_and_independent_of_state(tmp_path, raw_episode):
    later = tmp_path / "later"
    later.mkdir()
    with (raw_episode / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    original = payload["observations"]
    # Delay all measured movement by 0.5 seconds, without delaying commands.
    payload["observations"] = [original[max(0, index - 50)] for index in range(len(original))]
    with (later / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    for camera in pipeline.CAMERAS:
        (later / f"{camera}_image.mp4").symlink_to(raw_episode / f"{camera}_image.mp4")
    run = pipeline.RunConfig(min_remaining_seconds=1.)
    cutoffs = {}
    for source in (raw_episode, later):
        values = []
        for state in pipeline.STATES:
            for action in ("joint", "abs-ee"):
                setup = pipeline.Setup(state, action)
                aligned = pipeline.load_episode(source, setup, run)
                _, metric = pipeline.clip_episode(aligned, setup, run, 0)
                values.append(metric["removed_frames"])
        assert len(set(values)) == 1
        cutoffs[source] = values[0]
    assert cutoffs[later] - cutoffs[raw_episode] == 15


def test_full_pipeline_and_video_alignment(tmp_path, raw_episode, monkeypatch):
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", pipeline.resolve_ffmpeg(None))
    setup = pipeline.Setup("joint-last-command", "abs-ee")
    run = pipeline.RunConfig()
    output = tmp_path / "output"
    pipeline.convert(raw_episode.parent, output, "Test task.", setup, run, "custom")
    result = validate_dataset(output)
    assert result["passed"] and result["videos_checked"] == 2
    info = json.loads((output / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["action"]["shape"] == [8]
    _, _, _, _, indices = expected_from_raw(raw_episode, asdict(setup), asdict(run))
    for camera in pipeline.CAMERAS:
        original = cv2.VideoCapture(str(raw_episode / f"{camera}_image.mp4"))
        saved = cv2.VideoCapture(str(output / f"videos/chunk-000/observation.images.{camera}/episode_000000.mp4"))
        try:
            for frame_index in (0, len(indices[camera]) - 1):
                original.set(cv2.CAP_PROP_POS_FRAMES, int(indices[camera][frame_index]))
                saved.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok1, frame1 = original.read()
                ok2, frame2 = saved.read()
                assert ok1 and ok2
                assert np.max(np.abs(frame1.astype(float).mean(axis=(0, 1)) - frame2.astype(float).mean(axis=(0, 1)))) < 8
        finally:
            original.release()
            saved.release()
    with pytest.raises(FileExistsError):
        pipeline.convert(raw_episode.parent, output, "Test task.", setup, run, "custom")
    # A later validation failure must not replace an existing completed dataset.
    old_info = (output / "meta/info.json").read_bytes()
    import validate_piper_preprocessing as validator

    def fail(*args):
        raise AssertionError("intentional validation failure")
    monkeypatch.setattr(validator, "validate_dataset", fail)
    with pytest.raises(AssertionError, match="intentional"):
        pipeline.convert(raw_episode.parent, output, "Test task.", setup, run, "custom", overwrite=True)
    assert (output / "meta/info.json").read_bytes() == old_info
    assert not list(tmp_path.glob(".output_*"))


def test_cli_presets_and_dry_run(raw_episode, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["preprocess_piper.py", "--list-setups"])
    assert pipeline.main() == 0
    assert "state-abs-ee" in capsys.readouterr().out
    # Arrange task folder discovery without copying the source videos.
    root = tmp_path / "raw"
    (root / "hanging_mug").mkdir(parents=True)
    folder = root / "hanging_mug" / "episode"
    folder.mkdir()
    for path in raw_episode.iterdir():
        (folder / path.name).symlink_to(path)
    output = tmp_path / "not_created"
    monkeypatch.setattr(sys, "argv", ["preprocess_piper.py", "--all-setups", "--dry-run", "--raw-root", str(root), "--output-root", str(output)])
    assert pipeline.main() == 0
    text = capsys.readouterr().out
    assert text.count('"preset":') == len(pipeline.PRESETS)
    assert not output.exists()
