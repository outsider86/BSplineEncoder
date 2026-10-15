"""Semantic and end-to-end tests for the ``data`` preprocessing pipeline.

Ported from ``scripts/legacy_piper/test_preprocess_piper.py``; expected rows come
from the independent reconstruction in ``data.validation.dataset``.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")
pytest.importorskip("pyarrow")

from piper_fixtures import make_raw_episode  # noqa: E402
from data import (  # noqa: E402
    ACTIONS, PRESETS, STATES, ExtraObservation, PipelineConfig, RunConfig, Setup,
    SplineConfig, config_from_mapping, default_camera, run,
)
from data import pipeline  # noqa: E402
from data.transforms.cleanup import detect_cutoff, motion_signals  # noqa: E402
from data.config import CameraSpec  # noqa: E402
from data.conversion.episode import clip_episode, load_episode  # noqa: E402
from data.conversion.media import resolve_ffmpeg  # noqa: E402
from data.validation.dataset import expected_from_raw  # noqa: E402

CAMERAS = (default_camera("global"), default_camera("hand"))
CAMERA_DICTS = [asdict(camera) for camera in CAMERAS]


def _ffmpeg_or_skip() -> str:
    try:
        return resolve_ffmpeg(None)
    except RuntimeError:
        pytest.skip("ffmpeg unavailable")


@pytest.fixture(scope="module")
def raw_task(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw") / "classify_blocks"
    make_raw_episode(root / "ep_a")
    make_raw_episode(root / "ep_b", onset=1.6)
    return root


@pytest.fixture(scope="module")
def raw_episode(raw_task):
    return raw_task / "ep_a"


def _load(path, setup, run_config=None, cameras=CAMERAS, extras=()):
    run_config = run_config or RunConfig()
    aligned = load_episode(path, setup, run_config, cameras, extras)
    return aligned, *clip_episode(aligned, setup, run_config, 0)


COMBINATIONS = [
    Setup(state, action, offset, quaternion, clock)
    for state in STATES for action in ACTIONS
    for offset in ((1,) if action.startswith("delta-") else (0, 1))
    for quaternion in ("raw", "continuous") for clock in ("raw", "target")
]


@pytest.mark.parametrize("setup", COMBINATIONS, ids=lambda s: f"{s.state}-{s.action}-"
                         f"{s.action_offset}-{s.quaternion}-{s.last_command_clock}")
def test_all_state_action_combinations_against_raw(raw_episode, setup):
    _, episode, metric = _load(raw_episode, setup)
    state, action, cutoff, count, indices, _ = expected_from_raw(
        raw_episode, asdict(setup), asdict(RunConfig()), CAMERA_DICTS)
    np.testing.assert_array_equal(episode.state, state)
    np.testing.assert_allclose(episode.action, action, atol=2e-7, rtol=1e-6)
    assert metric["removed_frames"] == cutoff > 0
    assert episode.length == count - setup.action_offset - cutoff
    for camera in CAMERAS:
        np.testing.assert_array_equal(episode.camera_indices[camera.name], indices[camera.name])


@pytest.mark.parametrize("source", ("auto", "measured-joint", "measured-delta-joint",
                                    "command-joint", "measured-ee"))
@pytest.mark.parametrize("cleanup", (True, False))
def test_cleanup_override_and_disabled(raw_episode, source, cleanup):
    setup = Setup("last-command", "abs-ee")
    run_config = RunConfig(cleanup=cleanup, cleanup_motion=source, preroll_seconds=.1)
    _, episode, metric = _load(raw_episode, setup, run_config)
    state, action, cutoff, *_ = expected_from_raw(raw_episode, asdict(setup), asdict(run_config),
                                                  CAMERA_DICTS)
    assert metric["removed_frames"] == cutoff
    np.testing.assert_array_equal(episode.state, state)
    np.testing.assert_array_equal(episode.action, action)
    if not cleanup:
        assert cutoff == 0


def test_command_clocks_and_action_sources_are_distinct(raw_episode):
    run_config = RunConfig(cleanup=False)
    raw = _load(raw_episode, Setup("last-command", "joint"), run_config)[1]
    target = _load(raw_episode, Setup("last-command", "joint", last_command_clock="target"),
                   run_config)[1]
    assert np.any(raw.state != target.state)
    assert np.max(np.abs(raw.state - raw.action)) > 1


@pytest.mark.parametrize("frame", ("tool", "base"))
def test_delta_ee_reconstructs_next_pose_and_absolute_gripper(raw_episode, frame):
    setup = Setup("ee", "delta-ee", quaternion="continuous", delta_ee_frame=frame)
    aligned, *_ = _load(raw_episode, setup, RunConfig(cleanup=False))
    current, action = aligned.episode.state, aligned.episode.action
    rotation = Rotation.from_quat(current[:, 3:7])
    delta = Rotation.from_rotvec(action[:, 3:6])
    if frame == "tool":
        position = current[:, :3] + rotation.apply(action[:, :3])
        following = rotation * delta
    else:
        position = current[:, :3] + action[:, :3]
        following = delta * rotation
    np.testing.assert_allclose(position, aligned.measured_ee[1:, :3], atol=1e-7)
    expected = Rotation.from_quat(aligned.measured_ee[1:, 3:7])
    np.testing.assert_allclose((following.inv() * expected).magnitude(), 0, atol=2e-7)
    np.testing.assert_array_equal(action[:, 6], aligned.measured_ee[1:, 7])
    assert len(action) == len(aligned.measured_ee) - 1  # final row lacks a next target


def test_delta_joint_reconstructs_measured_next_joints_and_gripper(raw_episode):
    aligned, *_ = _load(raw_episode, PRESETS["state-delta-joint"][0], RunConfig(cleanup=False))
    state, action = aligned.episode.state, aligned.episode.action
    np.testing.assert_allclose(state[:, :6] + action[:, :6], aligned.measured_joint[1:, :6], atol=1e-7)
    np.testing.assert_array_equal(action[:, 6], aligned.measured_joint[1:, 6])
    assert np.any(action[:, 6] != aligned.measured_joint[1:, 6] - state[:, 6])
    assert len(state) == len(aligned.measured_joint) - 1


def test_quaternions_are_normalized_with_continuous_sign_only_when_requested(raw_episode):
    raw = _load(raw_episode, Setup("ee", "abs-ee"), RunConfig(cleanup=False))[1]
    continuous = _load(raw_episode, Setup("ee", "abs-ee", quaternion="continuous"),
                       RunConfig(cleanup=False))[1]
    assert np.any(np.einsum("ij,ij->i", raw.state[1:, 3:7], raw.state[:-1, 3:7]) < 0)
    dots = np.einsum("ij,ij->i", continuous.state[1:, 3:7], continuous.state[:-1, 3:7])
    assert np.all(dots > 0) and continuous.state[0, 6] >= 0
    np.testing.assert_allclose(np.linalg.norm(continuous.state[:, 3:7], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.abs(raw.state[:, 3:7]), np.abs(continuous.state[:, 3:7]), atol=1e-6)


@pytest.mark.parametrize("source", ("observation_delta_joint", "observation_delta_ee"))
def test_delta_cleanup_keeps_current_frame_that_initiates_motion(source):
    state = np.zeros((90, 7), np.float32)
    state[:, 6] = .3
    action = np.zeros((90, 7), np.float32)
    action[:, 6] = .3
    action[40:, 0] = .02
    config = RunConfig().detector()
    signals = motion_signals(state, action, config, "state", source)
    assert detect_cutoff(signals, len(state), config) == 30  # minimum 60 remaining
    config = RunConfig(min_remaining_seconds=1.).detector()
    signals = motion_signals(state, action, config, "state", source)
    assert detect_cutoff(signals, len(state), config) == 40


def test_joint_cleanup_keeps_first_changed_row():
    values = np.zeros((90, 7), np.float32)
    values[40:, 0] = np.arange(50) * .01
    config = RunConfig(min_remaining_seconds=1.).detector()
    signals = motion_signals(values, values, config, "state", "observation")
    cutoff = detect_cutoff(signals, len(values), config)
    assert cutoff == 41 and values[cutoff, 0] != values[cutoff - 1, 0] == 0


def test_all_static_and_short_episodes_keep_every_row(tmp_path):
    static = make_raw_episode(tmp_path / "static", static=True)
    for setup in (Setup("joint", "joint", 0), Setup("joint", "delta-joint"),
                  Setup("ee", "delta-ee"), Setup("joint", "command", 0)):
        aligned, episode, metric = _load(static, setup)
        assert metric["removed_frames"] == 0 and episode.length == aligned.episode.length
    short = make_raw_episode(tmp_path / "short", duration=0.3)
    aligned, episode, metric = _load(short, Setup("joint", "joint", 0))
    assert metric["removed_frames"] == 0 and episode.length == aligned.episode.length
    tiny = make_raw_episode(tmp_path / "tiny", duration=0.02, videos=False)
    with pytest.raises(ValueError, match="two target samples"):
        load_episode(tiny, Setup("joint", "joint"), RunConfig(), ())


def test_internal_pause_is_kept_and_crop_is_one_contiguous_suffix(tmp_path):
    path = make_raw_episode(tmp_path / "pause", onset=0.6, duration=4.0, pause=(1.5, 2.5))
    setup = Setup("joint", "joint", 0)
    aligned, episode, metric = _load(path, setup, RunConfig(min_remaining_seconds=1.))
    cutoff = metric["removed_frames"]
    assert 0 < cutoff < 30
    full = aligned.episode
    np.testing.assert_array_equal(episode.state, full.state[cutoff:])
    np.testing.assert_array_equal(episode.robot_indices, full.robot_indices[cutoff:])
    for name in episode.camera_indices:
        np.testing.assert_array_equal(episode.camera_indices[name], full.camera_indices[name][cutoff:])
    paused = np.all(np.diff(episode.state[:, :6], axis=0) == 0, axis=1)
    assert paused.sum() >= 25  # the one-second pause survives
    np.testing.assert_array_equal(episode.timestamps, np.arange(episode.length, dtype=np.float32) / 30)


def test_unselected_cameras_are_not_loaded_or_overlap_limiting(tmp_path):
    path = make_raw_episode(tmp_path / "subset")
    (path / "hand_image.mp4").unlink()  # unselected stream may be missing entirely
    only_global = (default_camera("global"),)
    aligned = load_episode(path, Setup("joint", "joint"), RunConfig(), only_global)
    assert set(aligned.episode.cameras) == {"global"}
    with pytest.raises(FileNotFoundError):
        load_episode(path, Setup("joint", "joint"), RunConfig(), CAMERAS)
    no_cameras = load_episode(path, Setup("joint", "joint"), RunConfig(), ())
    assert len(no_cameras.target_source_time) > len(aligned.target_source_time)  # robot-only overlap
    wrist = load_episode(path, Setup("joint", "joint"), RunConfig(),
                         (default_camera("global"), default_camera("wrist")))
    times = wrist.target_source_time
    assert times[0] >= 1.0 and times[-1] <= 2.0  # a selected short camera bounds the clock


def test_extra_observations_are_resampled_cropped_and_validated(raw_episode):
    extra = (ExtraObservation("observation.joint_velocity", "arm_joint_velocity"),)
    aligned, episode, metric = _load(raw_episode, Setup("joint", "joint"), extras=extra)
    values = episode.extras["observation.joint_velocity"]
    assert values.shape == (episode.length, 6)
    np.testing.assert_allclose(values[:, 0] * 70, episode.state[:, 0], atol=1e-6)
    with pytest.raises(ValueError, match="missing"):
        load_episode(raw_episode, Setup("joint", "joint"), RunConfig(), CAMERAS,
                     (ExtraObservation("observation.x", "not_a_field"),))
    with pytest.raises(ValueError, match="observation"):
        ExtraObservation("action.bad", "arm_pos").check()


def test_idle_cutoff_is_per_episode_and_independent_of_state(tmp_path, raw_episode):
    later = tmp_path / "later"
    later.mkdir()
    with (raw_episode / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    original = payload["observations"]
    payload["observations"] = [original[max(0, index - 50)] for index in range(len(original))]
    with (later / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    for name in ("global", "hand"):
        (later / f"{name}_image.mp4").symlink_to(raw_episode / f"{name}_image.mp4")
    run_config = RunConfig(min_remaining_seconds=1.)
    cutoffs = {}
    for source in (raw_episode, later):
        values = {(_load(source, Setup(state, action), run_config)[2]["removed_frames"])
                  for state in STATES for action in ("joint", "abs-ee")}
        assert len(values) == 1
        cutoffs[source] = values.pop()
    assert cutoffs[later] - cutoffs[raw_episode] == 15


def test_measured_only_recordings_need_no_commands(tmp_path, raw_episode):
    path = tmp_path / "measured_only"
    path.mkdir()
    with (raw_episode / "data.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    del payload["actions"]
    with (path / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    for name in ("global", "hand"):
        (path / f"{name}_image.mp4").symlink_to(raw_episode / f"{name}_image.mp4")
    assert load_episode(path, Setup("joint", "abs-ee"), RunConfig(), CAMERAS).command is None
    with pytest.raises(ValueError, match="commands"):
        load_episode(path, Setup("last-command", "abs-ee"), RunConfig(), CAMERAS)


@pytest.mark.parametrize("action", ("delta-joint", "delta-ee"))
def test_invalid_delta_offset_and_thresholds(action):
    with pytest.raises(ValueError, match="requires"):
        Setup("joint", action, 0).check()
    with pytest.raises(ValueError, match="positive"):
        RunConfig(fps=float("nan")).detector()


# --- End-to-end export ---------------------------------------------------------

def _config(raw_task, output_root, **changes) -> PipelineConfig:
    values = dict(raw_root=str(raw_task), output_root=str(output_root),
                  setups=("state-delta-joint",), jobs=2)
    values.update(changes)
    return PipelineConfig(**values)


@pytest.mark.parametrize("preset", ("state-delta-joint", "state-delta-ee", "state-abs-ee",
                                    "joint-last-command-command"))
def test_preset_pipeline_exports_valid_lerobot_v21(tmp_path, raw_task, preset):
    _ffmpeg_or_skip()
    [report] = run(_config(raw_task, tmp_path / "out", setups=(preset,)), log=lambda *_: None)
    dataset = Path(report["dataset"])
    assert dataset == tmp_path / "out" / PRESETS[preset][1] / "classify_blocks_30hz_cleanup"
    assert report["validation"]["passed"] and report["validation"]["videos_checked"] == 4
    info = json.loads((dataset / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_videos"] == 4
    assert len(list((dataset / "videos").rglob("*.mp4"))) == 4
    stats = [json.loads(line) for line in (dataset / "meta/episodes_stats.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in stats] == [0, 1]
    for key, feature in info["features"].items():
        assert key in stats[0]["stats"], key
        if feature["dtype"] == "video":
            assert np.asarray(stats[0]["stats"][key]["mean"]).shape == (3, 1, 1)
    assert not list(dataset.parent.glob(".classify*"))


@pytest.mark.parametrize("cameras", ((), ("hand",), ("global", "wrist")), ids=str)
def test_camera_subsets_and_key_mapping(tmp_path, raw_task, cameras):
    _ffmpeg_or_skip()
    specs = tuple(default_camera(name) for name in cameras)
    if cameras == ("hand",):
        specs = (CameraSpec("hand", "hand_image.mp4", "observation.images.wrist_top", "hand"),)
    [report] = run(_config(raw_task, tmp_path / "out", cameras=specs, check_videos=True),
                   log=lambda *_: None)
    dataset = Path(report["dataset"])
    info = json.loads((dataset / "meta/info.json").read_text())
    modality = json.loads((dataset / "meta/modality.json").read_text())
    videos = sorted(key for key, value in info["features"].items() if value["dtype"] == "video")
    assert videos == sorted(spec.key for spec in specs)
    assert sorted(v["original_key"] for v in modality["video"].values()) == videos
    assert info["total_videos"] == 2 * len(specs)
    assert len(list(dataset.rglob("*.mp4"))) == 2 * len(specs)
    assert (info["video_path"] is None) == (not specs)
    assert report["validation"]["videos_checked"] == 2 * len(specs)
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    import cv2

    for spec in specs:
        for record in episodes:
            path = dataset / info["video_path"].format(
                episode_chunk=0, video_key=spec.key, episode_index=record["episode_index"])
            capture = cv2.VideoCapture(str(path))
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == record["length"]
            capture.release()


def test_cleanup_alignment_and_provenance_are_consistent(tmp_path, raw_task):
    _ffmpeg_or_skip()
    import pyarrow.parquet as pq

    extras = (ExtraObservation("observation.joint_velocity", "arm_joint_velocity"),)
    [report] = run(_config(raw_task, tmp_path / "out", extra_observations=extras),
                   log=lambda *_: None)
    dataset = Path(report["dataset"])
    sources = [json.loads(line) for line in (dataset / "meta/source_episodes.jsonl").read_text().splitlines()]
    for index, source in enumerate(sources):
        data = pq.read_table(dataset / f"data/chunk-000/episode_{index:06d}.parquet")
        alignment = pq.read_table(dataset / f"meta/source_alignment/chunk-000/episode_{index:06d}.parquet")
        assert len(data) == len(alignment) == source["resampled_frames"] - 1 - source["removed_prefix_frames"]
        with (Path(source["source_episode"]) / "data.pkl").open("rb") as stream:
            payload = pickle.load(stream)
        robot = np.asarray(alignment["source_robot_index"])
        joints = np.stack([payload["observations"][i]["arm_joint_position"] for i in robot])
        np.testing.assert_allclose(np.asarray(data["observation.state"].to_pylist())[:, :6], joints, atol=1e-6)
        velocity = np.stack([payload["observations"][i]["arm_joint_velocity"] for i in robot])
        np.testing.assert_allclose(np.asarray(data["observation.joint_velocity"].to_pylist()), velocity, atol=1e-6)
        assert np.all(np.diff(np.asarray(alignment["source_time"])) > 0)
        assert np.asarray(alignment["robot_alignment_ms"]).max() <= 5.0 + 1e-6


def test_units_are_converted_once_and_validated(tmp_path, raw_task):
    _ffmpeg_or_skip()
    custom = Setup("ee", "delta-ee", quaternion="continuous", angle_unit="degree",
                   position_unit="millimeter")
    base = run(_config(raw_task, tmp_path / "a", setups=(), custom=replace(
        custom, angle_unit="radian", position_unit="meter"), cameras=()), log=lambda *_: None)
    converted = run(_config(raw_task, tmp_path / "b", setups=(), custom=custom, cameras=()),
                    log=lambda *_: None)
    import pyarrow.parquet as pq

    a = pq.read_table(Path(base[0]["dataset"]) / "data/chunk-000/episode_000000.parquet")
    b = pq.read_table(Path(converted[0]["dataset"]) / "data/chunk-000/episode_000000.parquet")
    action_a, action_b = (np.asarray(t["action"].to_pylist()) for t in (a, b))
    np.testing.assert_allclose(action_b[:, :3], action_a[:, :3] * 1000, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(action_b[:, 3:6], np.degrees(action_a[:, 3:6]), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(action_b[:, 6], action_a[:, 6])
    state_a, state_b = (np.asarray(t["observation.state"].to_pylist()) for t in (a, b))
    np.testing.assert_array_equal(state_b[:, 3:7], state_a[:, 3:7])  # quaternions unchanged
    modality = json.loads((Path(converted[0]["dataset"]) / "meta/modality.json").read_text())
    assert modality["action"]["rotation_rotvec"]["unit"] == "degree"


def test_overwrite_and_failure_preserve_completed_outputs(tmp_path, raw_task, monkeypatch):
    _ffmpeg_or_skip()
    config = _config(raw_task, tmp_path / "out", spline=SplineConfig(enabled=True))
    [report] = run(config, log=lambda *_: None)
    dataset, sidecar = Path(report["dataset"]), Path(report["sidecar"]["path"])
    before = {path: path.read_bytes() for path in (dataset / "meta/info.json",
                                                   sidecar / "manifest.json")}
    with pytest.raises(FileExistsError):
        run(config, log=lambda *_: None)
    from data.splines import sidecar as splines

    def fail(*args, **kwargs):
        raise RuntimeError("intentional sidecar failure")

    monkeypatch.setattr(splines, "encode_sidecar", fail)
    with pytest.raises(RuntimeError, match="intentional"):
        run(replace(config, overwrite=True), log=lambda *_: None)
    for path, content in before.items():
        assert path.read_bytes() == content
    assert not [p for p in dataset.parent.iterdir() if p.name.startswith(".")]


def test_overlapping_and_duplicate_paths_are_rejected(tmp_path, raw_task):
    with pytest.raises(ValueError, match="raw input"):
        pipeline.plan(_config(raw_task, raw_task / "out"))
    with pytest.raises(ValueError, match="raw input"):
        pipeline.plan(_config(raw_task, tmp_path, output=str(raw_task.parent)))
    with pytest.raises(ValueError, match="overlap"):
        pipeline.plan(_config(raw_task, tmp_path, output=str(tmp_path / "d"),
                              spline=SplineConfig(enabled=True, output=str(tmp_path / "d" / "s"))))
    with pytest.raises(ValueError, match="exactly one setup"):
        pipeline.plan(_config(raw_task, tmp_path, output=str(tmp_path / "d"),
                              setups=("state-abs-ee", "state-delta-joint")))
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.plan(_config(raw_task, tmp_path, setups=("state-abs-ee", "state-abs-ee")))


def test_task_folder_raw_root_allows_sibling_outputs(tmp_path):
    task = make_raw_episode(tmp_path / "classify_blocks" / "ep", videos=False).parent
    jobs = pipeline.plan(PipelineConfig(raw_root=str(task), output_root=str(tmp_path / "out"),
                                        setups=("state-abs-ee",), cameras=()))
    assert jobs[0].raw_task == task and jobs[0].task == "Classify the blocks."
    with pytest.raises(ValueError, match="raw input"):
        pipeline.plan(PipelineConfig(raw_root=str(task), output_root=str(task / "out"),
                                     setups=("state-abs-ee",), cameras=()))


def test_dry_run_writes_nothing_and_reports_plan(tmp_path, raw_task):
    output = tmp_path / "not_created"
    reports = run(_config(raw_task, output, setups=tuple(PRESETS), dry_run=True), log=lambda *_: None)
    assert len(reports) == len(PRESETS) and not output.exists()
    assert all(r["episodes"] == 2 and r["removed_frames"] > 0 for r in reports)
    limited = run(_config(raw_task, output, dry_run=True, limit_episodes=1), log=lambda *_: None)
    assert limited[0]["episodes"] == 1


def test_config_file_and_cli_overrides(tmp_path, raw_task, capsys):
    from data.cli import build_parser, config_from_args, main

    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "raw_root": str(raw_task), "output_root": str(tmp_path / "o"), "setups": ["state-abs-ee"],
        "cameras": {"global": "observation.images.top"}, "run": {"fps": 15.0},
        "extra_observations": {"observation.joint_velocity": "arm_joint_velocity"},
        "spline": {"enabled": True, "mode": "adaptive_left", "chunk_size": 8, "num_basis": 7},
    }))
    config = config_from_args(build_parser().parse_args(
        ["--config", str(path), "--target-fps", "10", "--action-offset", "0"]))
    assert config.run.fps == 10.0 and config.setup_overrides == {"action_offset": 0}
    assert config.cameras == (CameraSpec("global", "global_image.mp4", "observation.images.top", "global"),)
    assert config.spline.mode == "adaptive_left" and config.spline.enabled
    assert config_from_mapping(json.loads(path.read_text())).run.fps == 15.0
    assert main(["--config", str(path), "--dry-run"]) == 0
    assert not (tmp_path / "o").exists()
    with pytest.raises(ValueError, match="unknown pipeline config keys"):
        config_from_mapping({"raw_rot": "x"})
    assert main(["--list-setups"]) == 0
    assert "state-abs-ee" in capsys.readouterr().out
