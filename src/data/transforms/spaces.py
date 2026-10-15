"""State/action space transforms: quaternions, t->t+1 deltas, layouts, units.

Ported from ``convert_piper_to_lerobot_delta_ee.py`` and ``preprocess_piper.py``
(now in ``scripts/legacy_piper``) with identical numerical conventions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import Setup


def canonicalize_quaternions(quaternions: np.ndarray, label: str) -> np.ndarray:
    """Normalize XYZW quaternions and choose a continuous per-episode sign.

    The first sample gets a nonnegative ``w``; every later sample is flipped
    when its dot product with the previous sample is negative.
    """

    values = np.asarray(quaternions, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"{label}: expected quaternion shape (N, 4), got {values.shape}")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or np.any(norms < 1e-8):
        raise ValueError(f"{label}: quaternion contains NaN, Inf, or a near-zero norm")
    values /= norms[:, None]
    if values[0, 3] < 0:
        values[0] *= -1
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0:
            values[index] *= -1
    return values.astype(np.float32)

def next_step_ee_actions(measured_ee: np.ndarray, frame: str = "tool") -> np.ndarray:
    """Measured pose increment t->t+1 plus absolute measured gripper[t+1].

    ``frame="tool"`` (the historical convention): ``dp = R[t]^T (p[t+1]-p[t])``
    and ``dr = Log(R[t]^T R[t+1])``.  ``frame="base"``: ``dp = p[t+1]-p[t]``
    and ``dr = Log(R[t+1] R[t]^T)``.  Rotations are rotation vectors in radians.
    """

    if measured_ee.ndim != 2 or measured_ee.shape[1] != 8:
        raise ValueError(f"measured EE must have shape (N, 8), got {measured_ee.shape}")
    if len(measured_ee) < 2:
        raise ValueError("at least two measured EE samples are required")
    if frame not in ("tool", "base"):
        raise ValueError("delta EE frame must be 'tool' or 'base'")
    rotations = Rotation.from_quat(measured_ee[:, 3:7].astype(np.float64))
    current, following = rotations[:-1], rotations[1:]
    world_translation = (
        measured_ee[1:, :3].astype(np.float64) - measured_ee[:-1, :3].astype(np.float64)
    )
    if frame == "tool":
        translation = current.inv().apply(world_translation)
        rotvec = (current.inv() * following).as_rotvec()
    else:
        translation = world_translation
        rotvec = (following * current.inv()).as_rotvec()
    action = np.concatenate(
        [translation, rotvec, measured_ee[1:, 7:8].astype(np.float64)], axis=1
    ).astype(np.float32)
    if not np.isfinite(action).all():
        raise ValueError("constructed delta-EE action contains NaN or Inf")
    return action

def next_step_joint_actions(measured_joint: np.ndarray) -> np.ndarray:
    """Six measured joint increments ``q[t+1]-q[t]`` and absolute gripper[t+1]."""

    if measured_joint.ndim != 2 or measured_joint.shape[1] != 7 or len(measured_joint) < 2:
        raise ValueError("Measured joints must have shape (N, 7) with N >= 2")
    return np.concatenate(
        [np.diff(measured_joint[:, :6], axis=0), measured_joint[1:, 6:7]], axis=1
    ).astype(np.float32)


# --- Layout, units, and modality metadata ------------------------------------

def _blocks(setup: Setup, part: str) -> list[tuple[str, int, int, str, dict]]:
    """``(name, start, end, kind, extra)`` blocks; kind drives unit conversion."""

    angle = {"unit": setup.angle_unit}
    position = {"unit": setup.position_unit}
    blocks: list[tuple[str, int, int, str, dict]] = []
    if part == "state":
        prefix = 0
        if setup.state.startswith("joint"):
            blocks += [("measured_joints", 0, 6, "angle", angle),
                       ("measured_gripper", 6, 7, "gripper", {})]
            prefix = 7
        elif setup.state.startswith("ee"):
            quaternion = {"quaternion_order": "xyzw", "frame": "robot_base",
                          "quaternion_policy": setup.quaternion}
            blocks += [("measured_tcp_position", 0, 3, "position", {**position, "frame": "robot_base"}),
                       ("measured_tcp_quaternion_xyzw", 3, 7, "quaternion", quaternion),
                       ("measured_gripper", 7, 8, "gripper", {})]
            prefix = 8
        if "command" in setup.state:
            blocks += [("previous_command_joints", prefix, prefix + 6, "angle",
                        {**angle, "clock": setup.last_command_clock}),
                       ("previous_command_gripper", prefix + 6, prefix + 7, "gripper",
                        {"clock": setup.last_command_clock})]
        return blocks
    if setup.action in ("joint", "command"):
        return [("joints", 0, 6, "angle", angle), ("gripper", 6, 7, "gripper", {})]
    if setup.action == "delta-joint":
        return [("joints", 0, 6, "angle", {**angle, "absolute": False, "frame": "joint_space"}),
                ("gripper", 6, 7, "gripper", {})]
    if setup.action == "abs-ee":
        return [("position", 0, 3, "position", {**position, "frame": "robot_base"}),
                ("quaternion", 3, 7, "quaternion",
                 {"quaternion_order": "xyzw", "quaternion_policy": setup.quaternion}),
                ("gripper", 7, 8, "gripper", {})]
    frame = "measured_tcp_at_t" if setup.delta_ee_frame == "tool" else "robot_base"
    return [("translation", 0, 3, "position", {**position, "absolute": False, "frame": frame}),
            ("rotation_rotvec", 3, 6, "angle", {**angle, "absolute": False, "frame": frame}),
            ("gripper", 6, 7, "gripper", {})]

def apply_units(values: np.ndarray, setup: Setup, part: str) -> np.ndarray:
    """Convert exported radians/meters; identity for the default units."""

    result = np.asarray(values, dtype=np.float32)
    if setup.angle_unit == "radian" and setup.position_unit == "meter":
        return result
    result = result.astype(np.float64).copy()
    for _, start, end, kind, _ in _blocks(setup, part):
        if kind == "angle" and setup.angle_unit == "degree":
            result[:, start:end] = np.degrees(result[:, start:end])
        elif kind == "position" and setup.position_unit == "millimeter":
            result[:, start:end] *= 1000.0
    return result.astype(np.float32)

def modality(setup: Setup) -> tuple[dict, dict, str, str]:
    def block(key, start, end, extra):
        return {"original_key": key, "start": start, "end": end, "absolute": True,
                "dtype": "float32", **extra}

    state = {name: block("observation.state", start, end,
                         {"range": [0, 1]} if kind == "gripper" and not extra else extra)
             for name, start, end, kind, extra in _blocks(setup, "state")}
    for name, value in state.items():
        if "gripper" in name:
            value.setdefault("range", [0, 1])
    action = {name: block("action", start, end, extra)
              for name, start, end, kind, extra in _blocks(setup, "action")}
    action["gripper"].setdefault("range", [0, 1])
    layout = []
    if setup.state.startswith("joint"):
        layout.append("measured joints/gripper[t]")
    elif setup.state.startswith("ee"):
        layout.append("measured EE xyz/quaternion xyzw/gripper[t]")
    if "command" in setup.state:
        layout.append(f"previous command[{setup.last_command_clock} clock; first repeats first]")
    time = "t+1" if setup.action_offset else "t"
    if setup.action in ("joint", "command"):
        action_layout = f"{'measured' if setup.action == 'joint' else 'command'} joints/gripper[{time}]"
    elif setup.action == "delta-joint":
        action_layout = "measured joint delta q[t+1]-q[t] + absolute measured gripper[t+1]"
    elif setup.action == "abs-ee":
        action_layout = f"measured absolute EE xyz/quaternion xyzw/gripper[{time}]"
    elif setup.delta_ee_frame == "tool":
        action_layout = "measured local EE delta[t->t+1] + absolute measured gripper[t+1]"
    else:
        action_layout = "measured base-frame EE delta[t->t+1] + absolute measured gripper[t+1]"
    return state, action, " + ".join(layout), action_layout

def feature_names(setup: Setup, part: str) -> list[str]:
    names = []
    for name, start, end, _, _ in _blocks(setup, part):
        width = end - start
        names += [name] if width == 1 else [f"{name}_{i}" for i in range(width)]
    return names


__all__ = ["apply_units", "canonicalize_quaternions", "feature_names", "modality",
           "next_step_ee_actions", "next_step_joint_actions"]
