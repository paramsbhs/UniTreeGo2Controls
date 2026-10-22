"""Safety monitor (FR7, Phase 1 subset).

Checks every control cycle and latches critical faults: once tripped, the run
loop must stop issuing the nominal controller's torques and fall back to damping.
Estimator divergence, QP infeasibility and command timeouts are added in later
phases as those components appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from go2_wbc.robot import RobotIndex
from go2_wbc.sim.mujoco_env import SimState
from go2_wbc.utils.transforms import tilt_angle


@dataclass(frozen=True)
class Fault:
    time: float
    kind: str           # e.g. "tilt", "nonfinite_torque"
    critical: bool
    detail: str


@dataclass
class SafetyMonitor:
    idx: RobotIndex
    max_tilt: float = 0.6
    min_height: float = 0.10
    joint_limit_tol: float = 0.02
    faults: list[Fault] = field(default_factory=list)
    tripped: bool = False

    @classmethod
    def from_config(cls, idx: RobotIndex, ctrl_cfg: dict) -> SafetyMonitor:
        s = ctrl_cfg.get("safety", {})
        return cls(idx, max_tilt=s.get("max_tilt", 0.6), min_height=s.get("min_height", 0.10))

    def _emit(self, fault: Fault, new: list[Fault]) -> None:
        new.append(fault)
        self.faults.append(fault)
        if fault.critical:
            self.tripped = True

    def check(self, s: SimState, tau_cmd: np.ndarray) -> list[Fault]:
        new: list[Fault] = []
        t = s.time
        if not np.all(np.isfinite(tau_cmd)):
            self._emit(Fault(t, "nonfinite_torque", True, "NaN/inf in torque command"), new)
        if not (np.all(np.isfinite(s.q)) and np.all(np.isfinite(s.base_pos))):
            self._emit(Fault(t, "nonfinite_state", True, "NaN/inf in state"), new)
            return new
        tilt = tilt_angle(s.base_quat)
        if tilt > self.max_tilt:
            self._emit(Fault(t, "tilt", True, f"tilt {tilt:.2f} rad > {self.max_tilt}"), new)
        if s.base_pos[2] < self.min_height:
            self._emit(Fault(t, "low_base", True, f"base z {s.base_pos[2]:.3f} m"), new)
        lo, hi = self.idx.joint_range.T
        bad = (s.q < lo - self.joint_limit_tol) | (s.q > hi + self.joint_limit_tol)
        if bad.any():
            names = [self.idx.joint_names[i] for i in np.flatnonzero(bad)]
            self._emit(Fault(t, "joint_limit", True, f"outside range: {names}"), new)
        if np.any(np.abs(tau_cmd) > self.idx.torque_limit + 1e-9):
            self._emit(Fault(t, "torque_limit", False, "command exceeds actuator limit"), new)
        return new
