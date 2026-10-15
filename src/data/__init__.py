"""Robot data processing: raw Piper recordings -> LeRobot v2.1 -> spline sidecars.

Groups: ``conversion`` (raw recordings, video, LeRobot writer), ``transforms``
(state/action spaces, idle cleanup), ``validation`` (independent dataset and
sidecar checks) and ``splines`` (LeRobot action I/O and sidecar encoding).
``config``, ``pipeline`` and ``cli`` orchestrate one configurable run
(``spline-preprocess-piper`` / ``python -m data``).  This package may depend on
``spline_encoder``; the reverse is limited to lazy compatibility re-exports.
Video and dataset dependencies (OpenCV, ffmpeg, PyArrow, pandas) are imported
lazily, only when a run needs them.
"""

from .config import (
    ACTIONS, PRESETS, STATES, TASKS, CameraSpec, ExtraObservation, PipelineConfig, RunConfig,
    Setup, SplineConfig, config_from_mapping, default_camera, load_config,
)
from .pipeline import align_job, convert, dry_run, plan, run

__all__ = [
    "ACTIONS", "PRESETS", "STATES", "TASKS", "CameraSpec", "ExtraObservation", "PipelineConfig",
    "RunConfig", "Setup", "SplineConfig", "align_job", "config_from_mapping", "convert",
    "default_camera", "dry_run", "load_config", "plan", "run",
]
