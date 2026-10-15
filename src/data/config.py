"""Configuration for the Piper raw-recording -> LeRobot -> spline pipeline.

Defaults and preset names reproduce ``scripts/legacy_piper/preprocess_piper.py``.  A run
can be described entirely by one JSON file (TOML on Python >= 3.11); command
line flags override file values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .transforms.cleanup import CleanupConfig


STATES = ("joint", "joint-last-command", "last-command", "ee", "ee-last-command")
ACTIONS = ("joint", "command", "abs-ee", "delta-joint", "delta-ee")
STATE_SOURCE = {
    "joint": "state", "joint-last-command": "state_last_command",
    "last-command": "last_command", "ee": "observation_ee",
    "ee-last-command": "observation_ee_last_command",
}
ACTION_SOURCE = {
    "joint": "observation", "command": "action", "abs-ee": "observation_ee",
    "delta-joint": "observation_delta_joint", "delta-ee": "observation_delta_ee",
}
CLEANUP_MOTIONS = ("auto", "measured-joint", "measured-delta-joint", "command-joint", "measured-ee")
TASKS = {
    "classify_blocks": "Classify the blocks.",
    "hanging_mug": "Hang the mug.",
    "stacking_cup": "Stack the cups.",
}
_KEY = re.compile(r"^[A-Za-z0-9_.\-]+$")


@dataclass(frozen=True)
class Setup:
    """State/action space selection.

    ``action_offset`` 0 selects the action at ``t`` and 1 at ``t+1`` for
    absolute actions.  Deltas always mean ``t -> t+1`` and require offset 1.
    ``quaternion`` is ``raw`` (feedback as recorded) or ``continuous``
    (normalized XYZW with continuous sign).  ``last_command_clock`` selects
    the previous RAW robot sample (historical) or the previous target-clock
    sample for command-history state.  ``delta_ee_frame`` ``tool`` is the
    historical current-TCP frame; ``base`` expresses increments in the robot
    base frame.  ``angle_unit``/``position_unit`` convert exported joint
    angles and rotation vectors / positions; cleanup thresholds always use
    raw radians and meters.
    """

    state: str
    action: str
    action_offset: int = 1
    quaternion: str = "raw"
    last_command_clock: str = "raw"
    delta_ee_frame: str = "tool"
    angle_unit: str = "radian"
    position_unit: str = "meter"

    def check(self) -> None:
        if self.state not in STATES or self.action not in ACTIONS:
            raise ValueError(f"Unknown state/action: {self.state}/{self.action}")
        if self.action_offset not in (0, 1):
            raise ValueError("action-offset must be 0 or 1")
        if self.action.startswith("delta-") and self.action_offset != 1:
            raise ValueError(
                f"{self.action} requires --action-offset 1: measured state[t] -> state[t+1]"
            )
        if self.quaternion not in ("raw", "continuous"):
            raise ValueError("quaternion must be raw or continuous")
        if self.last_command_clock not in ("raw", "target"):
            raise ValueError("last-command-clock must be raw or target")
        if self.delta_ee_frame not in ("tool", "base"):
            raise ValueError("delta-ee-frame must be tool or base")
        if self.angle_unit not in ("radian", "degree"):
            raise ValueError("angle-unit must be radian or degree")
        if self.position_unit not in ("meter", "millimeter"):
            raise ValueError("position-unit must be meter or millimeter")

    @property
    def needs_ee(self) -> bool:
        return self.state.startswith("ee") or "ee" in self.action

    @property
    def needs_command(self) -> bool:
        return "command" in self.state or self.action == "command"


# Preset names and output families reproduce the historical dataset layouts.
PRESETS: dict[str, tuple[Setup, str]] = {
    "state-abs-ee": (Setup("joint", "abs-ee"), "StateAbsEE"),
    "state-abs-joint": (Setup("joint", "joint"), "StateAbsJoint"),
    "state-delta-joint": (Setup("joint", "delta-joint"), "StateDeltaJoint"),
    "state-delta-ee": (Setup("joint", "delta-ee", quaternion="continuous"), "StateDeltaEE"),
    "ee-delta-ee": (Setup("ee", "delta-ee", quaternion="continuous"), "ActionEE/State_EE"),
    "joint-command": (Setup("joint", "command", 0), "ActionCommand/State_Joint"),
    "joint-last-command-command": (Setup("joint-last-command", "command", 0),
                                   "ActionCommand/State_LastCommand_Joint"),
    "last-command-command": (Setup("last-command", "command", 0), "ActionCommand/LastCommand_Joint"),
    "joint-joint": (Setup("joint", "joint", 0), "ActionJoint/State_Joint"),
    "joint-last-command-joint": (Setup("joint-last-command", "joint", 0),
                                 "ActionJoint/State_LastCommand_Joint"),
    "last-command-joint": (Setup("last-command", "joint", 0), "ActionJoint/LastCommand_Joint"),
    "processed": (Setup("joint-last-command", "command", 0), "Processed"),
}


@dataclass(frozen=True)
class RunConfig:
    """Target clock and leading-idle cleanup (defaults match the source scripts)."""

    fps: float = 30.0
    cleanup: bool = True
    cleanup_motion: str = "auto"
    arm_threshold: float = 0.005
    gripper_threshold: float = 0.01
    translation_threshold: float = 0.0005
    rotation_threshold: float = 0.005
    confirmation_seconds: float = 0.4
    confirmation_fraction: float = 0.5
    preroll_seconds: float = 0.0
    max_scan_seconds: float = 10.0
    min_remaining_seconds: float = 2.0

    def detector(self) -> CleanupConfig:
        if self.cleanup_motion not in CLEANUP_MOTIONS:
            raise ValueError("Unknown cleanup motion source")
        values = (self.fps, self.arm_threshold, self.gripper_threshold,
                  self.translation_threshold, self.rotation_threshold,
                  self.confirmation_seconds, self.max_scan_seconds, self.min_remaining_seconds)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("FPS, thresholds and durations must be finite and positive")
        if not np.isfinite(self.preroll_seconds) or self.preroll_seconds < 0:
            raise ValueError("preroll-seconds must be finite and nonnegative")
        if not 0 < self.confirmation_fraction <= 1:
            raise ValueError("confirmation-fraction must be in (0, 1]")
        window = max(1, round(self.confirmation_seconds * self.fps))
        return CleanupConfig(
            self.arm_threshold, self.gripper_threshold,
            self.translation_threshold, self.rotation_threshold,
            window, max(1, round(window * self.confirmation_fraction)),
            round(self.preroll_seconds * self.fps), round(self.max_scan_seconds * self.fps),
            round(self.min_remaining_seconds * self.fps),
        )


@dataclass(frozen=True)
class CameraSpec:
    """One selected video stream.

    ``file`` is relative to each raw episode directory; ``timestamps_key``
    indexes ``camera_timestamps`` / ``camera_frame_source_timestamps`` in the
    raw payload; ``key`` is the LeRobot feature name.
    """

    name: str
    file: str
    key: str
    timestamps_key: str

    def check(self) -> None:
        if not self.name or not _KEY.match(self.name):
            raise ValueError(f"invalid camera name {self.name!r}")
        if not self.file or Path(self.file).name != self.file:
            raise ValueError(f"camera {self.name}: file must be one file name, got {self.file!r}")
        if not self.key or not _KEY.match(self.key) or "/" in self.key:
            raise ValueError(f"camera {self.name}: invalid LeRobot key {self.key!r}")


def default_camera(name: str, key: str | None = None) -> CameraSpec:
    return CameraSpec(name, f"{name}_image.mp4", key or f"observation.images.{name}", name)


DEFAULT_CAMERAS = (default_camera("global"), default_camera("hand"))


@dataclass(frozen=True)
class ExtraObservation:
    """A raw per-sample observation field exported as an extra float feature."""

    key: str
    field: str

    def check(self) -> None:
        if not _KEY.match(self.key) or not self.key.startswith("observation."):
            raise ValueError(f"extra observation key must start with 'observation.': {self.key!r}")
        if self.key in ("observation.state",) or self.key.startswith("observation.images."):
            raise ValueError(f"extra observation key {self.key!r} collides with a core feature")
        if not self.field:
            raise ValueError("extra observation field must be nonempty")


@dataclass(frozen=True)
class SplineConfig:
    """Optional B-spline sidecar over the FINAL cleaned ``action`` column.

    ``action_dim`` and ``frequency_hz`` always come from the exported
    dataset.  ``preset`` borrows geometry from a named spline preset;
    ``encoder_state`` reuses a saved encoder.  Either must be compatible with
    the dataset or the run is rejected.
    """

    enabled: bool = False
    mode: str = "uniform_left"
    preset: str | None = None
    encoder_state: str | None = None
    chunk_size: int = 16
    degree: int = 3
    num_basis: int = 11
    span_length_steps: int | None = 2
    vocab_size: int = 256
    regularization: float = 1e-4
    fit_tolerance: float = 0.01
    exclude_knot_dimensions: tuple[int, ...] = ()
    quantize: bool = False
    normalization: str = "none"
    validation_episodes: int = 0
    output: str | None = None

    def check(self) -> None:
        from spline_encoder.encoding.config import ENCODER_MODES

        if self.mode not in ENCODER_MODES:
            raise ValueError(f"spline mode must be one of {ENCODER_MODES}")
        if self.normalization not in ("none", "q01_q99"):
            raise ValueError("spline normalization must be none or q01_q99")
        if self.preset is not None and self.encoder_state is not None:
            raise ValueError("choose spline preset or encoder_state, not both")
        if not isinstance(self.validation_episodes, int) or self.validation_episodes < 0:
            raise ValueError("validation_episodes must be a nonnegative integer")


@dataclass(frozen=True)
class PipelineConfig:
    raw_root: str = "DATA"
    tasks: tuple[str, ...] = ()
    task_description: str | None = None
    output_root: str = "outputs/piper"
    output: str | None = None
    setups: tuple[str, ...] = ()
    setup_overrides: Mapping[str, Any] = field(default_factory=dict)
    custom: Setup | None = None
    run: RunConfig = field(default_factory=RunConfig)
    cameras: tuple[CameraSpec, ...] = DEFAULT_CAMERAS
    extra_observations: tuple[ExtraObservation, ...] = ()
    spline: SplineConfig = field(default_factory=SplineConfig)
    validate: bool = True
    check_videos: bool = True
    limit_episodes: int | None = None
    jobs: int = 1
    ffmpeg: str | None = None
    overwrite: bool = False
    dry_run: bool = False

    def check(self) -> None:
        self.run.detector()
        self.spline.check()
        if self.custom is not None:
            self.custom.check()
        for name in self.setups:
            if name not in PRESETS:
                raise ValueError(f"unknown setup {name!r}; choose from {tuple(PRESETS)}")
        if not self.setups and self.custom is None:
            raise ValueError("select at least one preset setup or a custom state/action")
        names = [camera.name for camera in self.cameras]
        keys = [camera.key for camera in self.cameras]
        if len(set(names)) != len(names) or len(set(keys)) != len(keys):
            raise ValueError("camera names and LeRobot keys must be unique")
        for camera in self.cameras:
            camera.check()
        extras = [extra.key for extra in self.extra_observations]
        if len(set(extras)) != len(extras) or set(extras) & set(keys):
            raise ValueError("extra observation keys must be unique and differ from cameras")
        for extra in self.extra_observations:
            extra.check()
        if self.limit_episodes is not None and self.limit_episodes < 1:
            raise ValueError("limit-episodes must be positive")
        if self.jobs < 1:
            raise ValueError("jobs must be positive")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Duplicate tasks")
        if self.task_description is not None and len(self.tasks) > 1:
            raise ValueError("task-description requires exactly one task")
        if self.output is not None and len(self.setups) + (self.custom is not None) * 1 > 1:
            raise ValueError("an explicit output path requires exactly one setup")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


def _tuple_of(cls, values) -> tuple:
    return tuple(value if isinstance(value, cls) else cls(**value) for value in values or ())


def _cameras(values) -> tuple[CameraSpec, ...]:
    """Accept ``["global", "hand"]``, ``{"global": "observation.images.top"}``
    or a list of full mappings."""

    if values is None:
        return DEFAULT_CAMERAS
    if isinstance(values, Mapping):
        return tuple(default_camera(name) if key is None else
                     (CameraSpec(**{"name": name, **key}) if isinstance(key, Mapping)
                      else default_camera(name, key))
                     for name, key in values.items())
    result = []
    for value in values:
        if isinstance(value, CameraSpec):
            result.append(value)
        elif isinstance(value, str):
            result.append(default_camera(value))
        else:
            spec = dict(value)
            name = spec["name"]
            result.append(CameraSpec(
                name=name, file=spec.get("file", f"{name}_image.mp4"),
                key=spec.get("key", f"observation.images.{name}"),
                timestamps_key=spec.get("timestamps_key", name),
            ))
    return tuple(result)


def config_from_mapping(values: Mapping[str, Any]) -> PipelineConfig:
    payload = dict(values)
    known = {item.name for item in fields(PipelineConfig)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown pipeline config keys: {sorted(unknown)}")
    if "run" in payload and not isinstance(payload["run"], RunConfig):
        payload["run"] = RunConfig(**payload["run"])
    if "spline" in payload and not isinstance(payload["spline"], SplineConfig):
        spline = dict(payload["spline"])
        spline["exclude_knot_dimensions"] = tuple(spline.get("exclude_knot_dimensions") or ())
        payload["spline"] = SplineConfig(**spline)
    if payload.get("custom") is not None and not isinstance(payload["custom"], Setup):
        payload["custom"] = Setup(**payload["custom"])
    if "cameras" in payload:
        payload["cameras"] = _cameras(payload["cameras"])
    if "extra_observations" in payload:
        extras = payload["extra_observations"]
        if isinstance(extras, Mapping):
            extras = [{"key": key, "field": raw} for key, raw in extras.items()]
        payload["extra_observations"] = _tuple_of(ExtraObservation, extras)
    for name in ("tasks", "setups"):
        if name in payload:
            payload[name] = tuple(payload[name] or ())
    return PipelineConfig(**payload)


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".toml":
        try:
            import tomllib
        except ImportError as exc:  # pragma: no cover - Python 3.10
            raise ValueError("TOML configs need Python >= 3.11; use JSON") from exc
        values = tomllib.loads(text)
    else:
        values = json.loads(text)
    if not isinstance(values, Mapping):
        raise ValueError(f"{path}: config root must be an object")
    return config_from_mapping(values)


def with_overrides(config: PipelineConfig, **changes: Any) -> PipelineConfig:
    return replace(config, **{key: value for key, value in changes.items() if value is not None})


__all__ = [
    "ACTIONS", "ACTION_SOURCE", "CLEANUP_MOTIONS", "CameraSpec", "DEFAULT_CAMERAS",
    "ExtraObservation", "PRESETS", "PipelineConfig", "RunConfig", "STATES", "STATE_SOURCE",
    "Setup", "SplineConfig", "TASKS", "config_from_mapping", "default_camera", "load_config",
    "with_overrides",
]
