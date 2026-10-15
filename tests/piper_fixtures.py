"""Synthetic raw Piper recordings shared by the preprocessing tests."""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation


def write_video(path: Path, count: int, shade: int, size=(64, 48)) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, size)
    assert writer.isOpened()
    for index in range(count):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:] = [index % 256, shade, 100]
        writer.write(frame)
    writer.release()


def make_raw_episode(path: Path, *, onset: float = 1.1, duration: float = 4.0,
                     pause: tuple[float, float] | None = None, static: bool = False,
                     wrist_window: tuple[float, float] = (1.0, 2.0),
                     videos: bool = True) -> Path:
    """Asynchronous robot (100 Hz) and camera clocks; motion starts at ``onset``.

    Quaternion signs are flipped on every third raw sample (same rotation)
    and command/FK EE fields are deliberately bogus so measured actions can
    never silently read them.
    """

    path.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(round(duration * 100)) + 1, dtype=np.float64) / 100.0
    motion = np.zeros_like(time) if static else np.maximum(0, time - onset)
    if pause is not None and not static:
        start, stop = pause
        motion = np.where(time < start, motion,
                          np.where(time < stop, start - onset, motion - (stop - start)))
    rotations = Rotation.from_euler(
        "xyz", np.column_stack([motion * .12, motion * .1, .8 + motion * .25]))
    quaternions = rotations.as_quat()
    quaternions[::3] *= -1
    observations, commands = [], []
    for index, t in enumerate(time):
        moving = (not static) and t >= onset + 0.15
        observations.append({
            "arm_joint_position": np.array([motion[index] * .7] * 6),
            "arm_joint_velocity": np.array([motion[index] * .01] * 6),
            "arm_pos": np.array([.2 + motion[index] * .1, .3 - motion[index] * .05, .4]),
            "arm_quat": quaternions[index],
            "gripper_pos": np.array([.7 if moving else .2]),
        })
        commands.append({
            "arm_joint_position": np.array([4 + (0 if static else max(0, t - .6))] * 6),
            "gripper_pos": np.array([.9 if (not static and t >= .8) else .1]),
            "arm_pos": np.array([99., 99., 99.]), "arm_quat": np.array([0., 0., 0., 1.]),
        })
    frames = int(duration * 30) - 1
    camera_times = {
        "global": np.arange(frames, dtype=np.float64) / 30.0 + .025,
        "hand": np.arange(frames - 4, dtype=np.float64) / 29.0 + .055,
        "wrist": np.arange(int((wrist_window[1] - wrist_window[0]) * 30), dtype=np.float64)
        / 30.0 + wrist_window[0],
    }
    if videos:
        for shade, (camera, timestamps) in zip((40, 150, 220), camera_times.items()):
            write_video(path / f"{camera}_image.mp4", len(timestamps), shade)
    payload = {"timestamps": time, "observations": observations, "actions": commands,
               "camera_timestamps": camera_times, "camera_frame_source_timestamps": camera_times}
    with (path / "data.pkl").open("wb") as stream:
        pickle.dump(payload, stream)
    return path
