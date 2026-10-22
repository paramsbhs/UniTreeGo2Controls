"""Joint-level PD control with saturation and joint-limit protection (FR6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go2_wbc.robot import RobotIndex


@dataclass
class JointCommandInfo:
    saturated: np.ndarray       # (12,) bool, torque was clipped
    q_des_clamped: np.ndarray   # (12,) bool, position target was pulled inside the soft limits


class JointPDController:
    """tau = s * (kp (q_des - q) + kd (dq_des - dq)) + tau_ff, then clipped to actuator limits.

    `s` is a 0..1 gain scale used for smooth startup. Position targets are clamped
    `margin` rad inside the joint range so the PD never drives into a hard stop.
    """

    def __init__(self, idx: RobotIndex, kp: np.ndarray, kd: np.ndarray, margin: float = 0.05):
        self.idx = idx
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        lo, hi = idx.joint_range.T
        self.q_min, self.q_max = lo + margin, hi - margin
        self.tau_max = idx.torque_limit

    @classmethod
    def from_config(cls, idx: RobotIndex, ctrl_cfg: dict) -> JointPDController:
        g = ctrl_cfg["gains"]
        kp = np.array([g[idx.joint_type(i)]["kp"] for i in range(idx.n_joints)])
        kd = np.array([g[idx.joint_type(i)]["kd"] for i in range(idx.n_joints)])
        return cls(idx, kp, kd, ctrl_cfg.get("joint_limit_margin", 0.05))

    def compute(
        self,
        q_des: np.ndarray,
        q: np.ndarray,
        dq: np.ndarray,
        dq_des: np.ndarray | float = 0.0,
        tau_ff: np.ndarray | float = 0.0,
        gain_scale: float = 1.0,
    ) -> tuple[np.ndarray, JointCommandInfo]:
        q_clamped = np.clip(q_des, self.q_min, self.q_max)
        tau = gain_scale * (self.kp * (q_clamped - q) + self.kd * (dq_des - dq)) + tau_ff
        tau_out = np.clip(tau, -self.tau_max, self.tau_max)
        info = JointCommandInfo(
            saturated=np.abs(tau) > self.tau_max,
            q_des_clamped=q_clamped != q_des,
        )
        return tau_out, info

    def damping(self, dq: np.ndarray, kd: float = 3.0) -> np.ndarray:
        """Pure joint damping: the safe fallback after a critical fault."""
        return np.clip(-kd * dq, -self.tau_max, self.tau_max)
