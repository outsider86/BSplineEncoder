"""Optional video I/O (OpenCV + ffmpeg), imported lazily.

The ffmpeg command is identical to ``scripts/legacy_piper/convert_piper_to_lerobot.py``.
Image statistics follow LeRobot v2.1 ``compute_episode_stats``: sampled
frames, auto-downsampled, RGB channels scaled to ``[0, 1]``, per-channel
min/max/mean/std with shape ``(3, 1, 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


@dataclass(frozen=True)
class CameraStream:
    name: str
    path: Path
    timestamps: np.ndarray
    capture_timestamps: np.ndarray
    width: int
    height: int
    source_fps: float


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "video streams need OpenCV: `pip install spline-encoder[piper]` "
            "or select no cameras"
        ) from exc
    return cv2


def video_properties(path: Path) -> tuple[int, int, int, float]:
    cv2 = _cv2()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
    )
    capture.release()
    if result[0] <= 0 or result[1] <= 0 or result[2] <= 0 or result[3] <= 0:
        raise ValueError(f"{path}: invalid video metadata {result}")
    return result


def resolve_ffmpeg(explicit: str | Path | None = None) -> str:
    """Explicit path, ``IMAGEIO_FFMPEG_EXE``, ``PATH``, imageio-ffmpeg, then a
    conda-base imageio-ffmpeg binary (as in the source scripts)."""

    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--ffmpeg does not exist: {path}")
        return str(path)
    configured = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if configured and Path(configured).is_file():
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    base = Path(sys.prefix).parent.parent
    for path in sorted(base.glob("lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg*")):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    raise RuntimeError("ffmpeg is required; install it, set IMAGEIO_FFMPEG_EXE, or pass --ffmpeg")


def stats_sample_indices(length: int, minimum: int = 100, maximum: int = 10_000,
                         power: float = 0.75) -> np.ndarray:
    """LeRobot ``sample_indices``: about ``length**0.75`` evenly spaced frames."""

    if length < minimum:
        count = length
    else:
        count = max(minimum, min(int(length**power), maximum))
    return np.round(np.linspace(0, length - 1, count)).astype(int)


def _downsample(frame: np.ndarray, target: int = 150, threshold: int = 300) -> np.ndarray:
    height, width = frame.shape[:2]
    if max(width, height) < threshold:
        return frame
    factor = int(width / target) if width > height else int(height / target)
    return frame[::factor, ::factor]


class _ImageStats:
    def __init__(self, wanted: np.ndarray):
        self.wanted = set(int(index) for index in wanted)
        self.frames: list[np.ndarray] = []

    def add(self, position: int, frame_bgr: np.ndarray) -> None:
        if position in self.wanted:
            rgb = _downsample(frame_bgr)[..., ::-1].astype(np.float64) / 255.0
            self.frames.append(rgb.reshape(-1, 3))

    def result(self) -> dict[str, list]:
        values = np.concatenate(self.frames, axis=0)

        def channel(stat):
            return [[[float(v)]] for v in stat]

        return {
            "min": channel(values.min(axis=0)), "max": channel(values.max(axis=0)),
            "mean": channel(values.mean(axis=0)), "std": channel(values.std(axis=0)),
            "count": [len(self.frames)],
        }


def write_resampled_video(stream: CameraStream, indices: np.ndarray, destination: Path,
                          target_fps: float, ffmpeg: str) -> dict[str, list]:
    """Encode the selected source frames at the target clock; return image stats."""

    cv2 = _cv2()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{stream.width}x{stream.height}", "-r", str(target_fps),
        "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE)
    capture = cv2.VideoCapture(str(stream.path))
    stats = _ImageStats(stats_sample_indices(len(indices)))
    if not capture.isOpened():
        process.kill()
        process.wait()
        raise ValueError(f"Could not open {stream.path}")
    assert process.stdin is not None and process.stderr is not None
    source_index = -1
    frame = None
    try:
        for position, requested in enumerate(indices):
            requested = int(requested)
            while source_index < requested:
                ok, frame = capture.read()
                source_index += 1
                if not ok or frame is None:
                    raise ValueError(f"{stream.path}: failed to decode source frame {source_index}")
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            stats.add(position, frame)
        process.stdin.close()
        error_text = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {destination}: {error_text.strip()}")
        width, height, frames, fps = video_properties(destination)
        if (width, height) != (stream.width, stream.height):
            raise RuntimeError(f"{destination}: encoded resolution {(width, height)} differs")
        if frames != len(indices) or not np.isclose(fps, target_fps, atol=1e-3):
            raise RuntimeError(f"{destination}: expected {len(indices)} frames at "
                               f"{target_fps:g} Hz, got {frames} at {fps:g} Hz")
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        destination.unlink(missing_ok=True)
        raise
    finally:
        capture.release()
    return stats.result()


__all__ = ["CameraStream", "resolve_ffmpeg", "stats_sample_indices", "video_properties",
           "write_resampled_video"]
