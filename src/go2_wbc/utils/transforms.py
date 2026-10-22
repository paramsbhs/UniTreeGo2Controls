"""Rotation and quaternion helpers.

Convention used everywhere in this repo: quaternions are scalar-first [w, x, y, z]
(MuJoCo's convention). Pinocchio uses scalar-last [x, y, z, w]; convert only at the
Pinocchio boundary with `wxyz_to_xyzw` / `xyzw_to_wxyz`.
Euler angles (roll, pitch, yaw; ZYX intrinsic) are for logging and plots only.
"""

from __future__ import annotations

import numpy as np


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 1:4], q[..., :1]], axis=-1)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] -> [roll, pitch, yaw] (ZYX intrinsic). For logging only."""
    w, x, y, z = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """[roll, pitch, yaw] (ZYX intrinsic) -> [w, x, y, z]."""
    r, p, y = np.asarray(rpy, dtype=float) / 2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def tilt_angle(q: np.ndarray) -> float:
    """Angle between the body z-axis and world z-axis (rad). Yaw-independent."""
    R = quat_to_rotmat(q)
    return float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
