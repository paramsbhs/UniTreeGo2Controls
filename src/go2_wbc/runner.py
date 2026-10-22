"""Shared closed-loop run logic for the stance controller.

Scripts and tests both call `run_stance`, so what the tests verify is exactly
what the scripts run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from go2_wbc.control.safety_monitor import SafetyMonitor
from go2_wbc.control.stance_controller import StanceController
from go2_wbc.sim.mujoco_env import Go2Env, SimState
from go2_wbc.utils.logging import RunLogger
from go2_wbc.utils.transforms import tilt_angle

HeightSchedule = Callable[[float], float]


@dataclass
class RunSummary:
    seed: int
    duration: float
    fell: bool
    first_fault: str | None
    height_rms_error: float      # m, over the settled part of the run
    max_tilt: float              # rad
    saturation_fraction: float   # fraction of control steps with any torque clipped
    wall_time: float             # s
    realtime_factor: float

    def report(self) -> str:
        why = f"  ({self.first_fault})" if self.fell else ""
        return (
            f"duration        {self.duration:8.2f} s\n"
            f"fell            {self.fell!s:>8}{why}\n"
            f"height RMS err  {self.height_rms_error * 1000:8.2f} mm\n"
            f"max tilt        {np.degrees(self.max_tilt):8.2f} deg\n"
            f"saturated steps {self.saturation_fraction * 100:8.2f} %\n"
            f"realtime factor {self.realtime_factor:8.1f} x"
        )


def run_stance(
    env: Go2Env,
    ctrl: StanceController,
    duration: float,
    seed: int | None = None,
    height_schedule: HeightSchedule | None = None,
    logger: RunLogger | None = None,
    monitor: SafetyMonitor | None = None,
    viewer=None,
    realtime: bool = False,
    settle_time: float = 1.0,
    on_step: Callable[[SimState], None] | None = None,
) -> RunSummary:
    seed = env.cfg["sim"].get("seed", 0) if seed is None else seed
    s = env.reset(seed)
    ctrl.reset(t0=s.time)
    monitor = monitor or SafetyMonitor.from_config(env.idx, env.cfg["controller"])

    n_steps = int(round(duration / env.control_dt))
    h_err, tilts, sat = [], [], 0
    wall0 = time.perf_counter()

    for _ in range(n_steps):
        tick = time.perf_counter()
        if height_schedule is not None:
            ctrl.set_height(height_schedule(s.time))

        tau, info = ctrl.update(s)
        monitor.check(s, tau)
        if monitor.tripped:
            tau = ctrl.pd.damping(s.dq)          # latched fallback, never resumes
        sat += info["n_saturated"] > 0

        if logger is not None:
            logger.log(s, tau, h_ref=info["h_ref"], h_int=info["h_int"],
                       startup_alpha=info["startup_alpha"], fault=float(monitor.tripped))

        s = env.step(tau)
        if on_step is not None:
            on_step(s)
        if s.time >= settle_time:
            h_err.append(ctrl.h_ref - s.base_pos[2])
        tilts.append(tilt_angle(s.base_quat))

        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
        if realtime:
            left = env.control_dt - (time.perf_counter() - tick)
            if left > 0:
                time.sleep(left)

    wall = time.perf_counter() - wall0
    crit = [f for f in monitor.faults if f.critical]
    return RunSummary(
        seed=seed,
        duration=s.time,
        fell=monitor.tripped,
        first_fault=f"{crit[0].kind} at t={crit[0].time:.2f}s: {crit[0].detail}" if crit else None,
        height_rms_error=float(np.sqrt(np.mean(np.square(h_err)))) if h_err else float("nan"),
        max_tilt=float(max(tilts)) if tilts else 0.0,
        saturation_fraction=sat / max(n_steps, 1),
        wall_time=wall,
        realtime_factor=s.time / wall if wall > 0 else float("inf"),
    )


def build(cfg: dict) -> tuple[Go2Env, StanceController]:
    env = Go2Env(cfg)
    ctrl = StanceController(env.idx, cfg["robot"], cfg["controller"], env.control_dt)
    return env, ctrl
