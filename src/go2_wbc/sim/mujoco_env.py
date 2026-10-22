"""Deterministic MuJoCo environment for the Unitree Go2.

The environment owns the physics. It exposes:
  * reset(seed)       -> SimState   (bit-exact for the same seed)
  * step(tau)         -> SimState   (tau in canonical joint order, N*m)
  * state()           -> SimState   (ground truth + raw IMU readings)

Everything a controller or estimator needs goes through SimState, so later phases
can swap ground truth for estimated quantities without touching the sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from go2_wbc.config import resolve
from go2_wbc.robot import RobotIndex
from go2_wbc.utils.transforms import quat_to_rotmat


@dataclass
class SimState:
    """Snapshot of the simulator. All arrays are copies, safe to store."""

    time: float
    # Ground truth (for evaluation and for Phase 1 control)
    base_pos: np.ndarray            # (3,) world frame, m
    base_quat: np.ndarray           # (4,) [w, x, y, z], body -> world
    base_lin_vel: np.ndarray        # (3,) world frame, m/s
    base_ang_vel: np.ndarray        # (3,) BODY frame, rad/s
    q: np.ndarray                   # (12,) canonical order, rad
    dq: np.ndarray                  # (12,) rad/s
    tau: np.ndarray                 # (12,) torque actually applied (after ctrlrange clamp)
    foot_pos: np.ndarray            # (4, 3) foot-sphere centres, world frame
    foot_force: np.ndarray          # (4,) total normal contact force per foot, N
    foot_contact: np.ndarray        # (4,) bool, foot_force > threshold
    # Raw sensors (what a real robot would have)
    imu_gyro: np.ndarray            # (3,) body-frame angular velocity, rad/s
    imu_acc: np.ndarray             # (3,) specific force in IMU frame, m/s^2
    extras: dict = field(default_factory=dict)

    @property
    def base_lin_vel_body(self) -> np.ndarray:
        return quat_to_rotmat(self.base_quat).T @ self.base_lin_vel


class Go2Env:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        rcfg, scfg = cfg["robot"], cfg["sim"]

        spec = mujoco.MjSpec.from_file(str(resolve(rcfg["scene"])))
        self._add_imu_sensors(spec, rcfg["imu_site"])
        self.model = spec.compile()
        self.model.opt.timestep = float(scfg["timestep"])
        self.data = mujoco.MjData(self.model)

        self.idx = RobotIndex.from_model(self.model, rcfg)
        self.decimation = int(scfg.get("control_decimation", 1))
        self.contact_threshold = float(rcfg["contact"]["force_threshold"])

        self._free_qpos = self._free_joint_qposadr()
        self._free_dof = self.model.jnt_dofadr[self._free_joint_id]
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, rcfg["home_keyframe"])
        if key < 0:
            raise ValueError(f"keyframe {rcfg['home_keyframe']!r} not found")
        self._home_key = key
        self._gyro_adr = self._sensor_adr("imu_gyro")
        self._acc_adr = self._sensor_adr("imu_acc")

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _add_imu_sensors(spec: mujoco.MjSpec, site: str) -> None:
        """Add gyro + accelerometer at the IMU site (the Menagerie model has no sensors)."""
        for name, typ in (("imu_gyro", mujoco.mjtSensor.mjSENS_GYRO),
                          ("imu_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
            spec.add_sensor(name=name, type=typ, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site)

    def _free_joint_qposadr(self) -> int:
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("expected exactly one free joint (the floating base)")
        self._free_joint_id = int(free[0])
        return int(self.model.jnt_qposadr[free[0]])

    def _sensor_adr(self, name: str) -> slice:
        s = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr, dim = self.model.sensor_adr[s], self.model.sensor_dim[s]
        return slice(adr, adr + dim)

    # -------------------------------------------------------------- properties
    @property
    def sim_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation

    # ---------------------------------------------------------------- control
    def reset(self, seed: int | None = None) -> SimState:
        """Reset to the home keyframe plus seeded noise. Same seed -> identical state."""
        seed = self.cfg["sim"].get("seed", 0) if seed is None else seed
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data

        mujoco.mj_resetDataKeyframe(m, d, self._home_key)
        noise = self.cfg["sim"].get("init_noise", {})
        jstd = float(noise.get("joint_pos_std", 0.0))
        hstd = float(noise.get("base_height_std", 0.0))
        if jstd > 0:
            q = d.qpos[self.idx.qpos_adr] + rng.normal(0.0, jstd, self.idx.n_joints)
            lo, hi = self.idx.joint_range.T
            d.qpos[self.idx.qpos_adr] = np.clip(q, lo, hi)
        if hstd > 0:
            d.qpos[self._free_qpos + 2] += rng.normal(0.0, hstd)
        d.ctrl[:] = 0.0
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(m, d)
        return self.state()

    def step(self, tau: np.ndarray) -> SimState:
        """Apply joint torques (canonical order) for one control period."""
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.idx.n_joints,):
            raise ValueError(f"tau must have shape ({self.idx.n_joints},), got {tau.shape}")
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError("non-finite torque command; refusing to step")
        self.data.ctrl[self.idx.actuator_ids] = tau
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def set_base_force(self, force_world: np.ndarray) -> None:
        """Constant external force on the base COM (world frame). Zero it to release."""
        self.data.xfrc_applied[self.idx.base_body_id, :3] = force_world

    # ------------------------------------------------------------------ state
    def foot_contact_forces(self) -> np.ndarray:
        """Total normal force between each foot and the floor."""
        m, d = self.model, self.data
        forces = np.zeros(len(self.idx.foot_geom_ids))
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if self.idx.floor_geom_id not in pair:
                continue
            for k, g in enumerate(self.idx.foot_geom_ids):
                if int(g) in pair:
                    mujoco.mj_contactForce(m, d, i, f6)
                    forces[k] += f6[0]  # contact frame x-axis is the normal
        return forces

    def state(self) -> SimState:
        d, idx = self.data, self.idx
        fp, fd = self._free_qpos, self._free_dof
        forces = self.foot_contact_forces()
        return SimState(
            time=float(d.time),
            base_pos=d.qpos[fp : fp + 3].copy(),
            base_quat=d.qpos[fp + 3 : fp + 7].copy(),
            base_lin_vel=d.qvel[fd : fd + 3].copy(),
            base_ang_vel=d.qvel[fd + 3 : fd + 6].copy(),
            q=d.qpos[idx.qpos_adr].copy(),
            dq=d.qvel[idx.dof_adr].copy(),
            tau=d.actuator_force[idx.actuator_ids].copy(),
            foot_pos=d.geom_xpos[idx.foot_geom_ids].copy(),
            foot_force=forces,
            foot_contact=forces > self.contact_threshold,
            imu_gyro=d.sensordata[self._gyro_adr].copy(),
            imu_acc=d.sensordata[self._acc_adr].copy(),
        )
