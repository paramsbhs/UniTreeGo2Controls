"""Phase 1 stance controller: regulate body height with a joint-space PD.

Each leg is treated as a planar 2-link chain (thigh + calf) with the foot directly
below the hip-pitch axis. A desired leg length gives thigh and knee angles through
the law of cosines; an integral loop on measured base height removes the
steady-state sag a pure PD leaves under gravity.

This is deliberately simple. It is the baseline the QP whole-body controller
(Phase 4) has to beat, and the fallback the safety monitor drops back to.
"""

from __future__ import annotations

import numpy as np

from go2_wbc.control.joint_controller import JointPDController
from go2_wbc.robot import RobotIndex
from go2_wbc.sim.mujoco_env import SimState


def leg_ik_vertical(length: float, l1: float, l2: float) -> tuple[float, float]:
    """Thigh and knee angle that place the foot `length` m straight below the hip axis.

    Go2 sign convention: positive thigh pitches the leg backwards-down, knee is negative.
    """
    length = float(np.clip(length, abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6))
    # interior knee angle
    cos_k = (l1**2 + l2**2 - length**2) / (2 * l1 * l2)
    knee_interior = np.arccos(np.clip(cos_k, -1.0, 1.0))
    knee = -(np.pi - knee_interior)
    # angle between thigh and the hip->foot line
    cos_a = (l1**2 + length**2 - l2**2) / (2 * l1 * length)
    thigh = np.arccos(np.clip(cos_a, -1.0, 1.0))
    return float(thigh), float(knee)


class StanceController:
    def __init__(self, idx: RobotIndex, robot_cfg: dict, ctrl_cfg: dict, dt: float):
        self.idx = idx
        self.dt = dt
        geo = robot_cfg["geometry"]
        self.l1, self.l2 = geo["thigh_length"], geo["calf_length"]
        self.foot_r = geo["foot_radius"]

        self.pd = JointPDController.from_config(idx, ctrl_cfg)
        self.h_min, self.h_max = ctrl_cfg["height_limits"]
        self.h_rate = ctrl_cfg["height_rate_limit"]
        self.ki = ctrl_cfg["height_ki"]
        self.i_lim = ctrl_cfg["height_i_limit"]
        self.ramp = ctrl_cfg["startup_ramp"]
        self.nominal = ctrl_cfg["nominal_height"]
        self.use_ff = bool(ctrl_cfg.get("static_load_ff", True))
        self.leg_force = robot_cfg["mass"] * 9.81 / len(idx.legs)  # N per leg, equal share
        self.reset()

    def reset(self, t0: float = 0.0) -> None:
        self.t0 = t0
        self.h_target = self.nominal   # what the user asked for
        self.h_ref = self.nominal      # rate-limited setpoint actually tracked
        self.h_int = 0.0
        self.q_start: np.ndarray | None = None  # measured pose at the first update

    def set_height(self, h: float) -> None:
        self.h_target = float(np.clip(h, self.h_min, self.h_max))

    def joint_targets(self, base_height: float) -> np.ndarray:
        leg_len = base_height - self.foot_r
        thigh, knee = leg_ik_vertical(leg_len, self.l1, self.l2)
        q = np.zeros(self.idx.n_joints)
        for i in range(self.idx.n_joints):
            t = self.idx.joint_type(i)
            q[i] = 0.0 if t == "hip" else thigh if t == "thigh" else knee
        return q

    def static_load_ff(self, q: np.ndarray) -> np.ndarray:
        """Knee torque that holds an equal share of body weight: tau = J^T f.

        With the foot under the hip and a vertical ground force F per leg, the
        foot's moment arm about the knee is l1*sin(thigh); the hip-pitch arm is ~0.
        Positive knee torque extends the leg (pushes the body up).
        """
        tau = np.zeros(self.idx.n_joints)
        thighs = {self.idx.joint_names[i].split("_")[0]: q[i]
                  for i in range(self.idx.n_joints) if self.idx.joint_type(i) == "thigh"}
        for i in range(self.idx.n_joints):
            if self.idx.joint_type(i) == "calf":
                leg = self.idx.joint_names[i].split("_")[0]
                tau[i] = self.leg_force * self.l1 * np.sin(thighs[leg])
        return tau

    def update(self, s: SimState) -> tuple[np.ndarray, dict]:
        # rate-limit the height setpoint
        step = np.clip(self.h_target - self.h_ref, -self.h_rate * self.dt, self.h_rate * self.dt)
        self.h_ref += step
        # integral on height error (ground-truth height in Phase 1; estimator later)
        err = self.h_ref - s.base_pos[2]
        self.h_int = float(np.clip(self.h_int + self.ki * err * self.dt, -self.i_lim, self.i_lim))

        q_des = self.joint_targets(self.h_ref + self.h_int)

        # Smooth startup: full gains, but the target blends from the measured start
        # pose with a min-jerk profile. (Ramping the *gains* instead lets the body
        # sag under gravity before the PD catches it - see docs/failures.md.)
        if self.q_start is None:
            self.q_start = s.q.copy()
        tau_r = (s.time - self.t0) / self.ramp if self.ramp > 0 else 1.0
        alpha = 1.0 if tau_r >= 1.0 else 10 * tau_r**3 - 15 * tau_r**4 + 6 * tau_r**5
        q_des = self.q_start + alpha * (q_des - self.q_start)

        tau_ff = self.static_load_ff(s.q) * alpha if self.use_ff else 0.0
        tau, info = self.pd.compute(q_des, s.q, s.dq, tau_ff=tau_ff)
        return tau, {
            "q_des": q_des,
            "h_ref": self.h_ref,
            "h_int": self.h_int,
            "startup_alpha": alpha,
            "n_saturated": int(info.saturated.sum()),
        }
