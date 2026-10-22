"""Explicit index maps between the canonical joint order and the MuJoCo model.

Nothing else in the codebase should index d.qpos / d.qvel / d.ctrl with
hard-coded offsets. Use RobotIndex instead, so a reordered or modified MJCF
breaks loudly here rather than silently in the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

JOINT_TYPES = ("hip", "thigh", "calf")


@dataclass(frozen=True)
class RobotIndex:
    joint_names: tuple[str, ...]
    legs: tuple[str, ...]
    qpos_adr: np.ndarray        # (12,) index into d.qpos for each canonical joint
    dof_adr: np.ndarray         # (12,) index into d.qvel / qfrc for each canonical joint
    actuator_ids: np.ndarray    # (12,) actuator driving each canonical joint
    joint_range: np.ndarray     # (12, 2) rad
    torque_limit: np.ndarray    # (12,) N*m, symmetric, from actuator ctrlrange
    foot_geom_ids: np.ndarray   # (4,)
    floor_geom_id: int
    imu_site_id: int
    base_body_id: int

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def joint_type(self, i: int) -> str:
        name = self.joint_names[i]
        for t in JOINT_TYPES:
            if f"_{t}_" in name:
                return t
        raise ValueError(f"cannot infer joint type of {name!r}")

    @classmethod
    def from_model(cls, m: mujoco.MjModel, robot_cfg: dict) -> RobotIndex:
        names = tuple(robot_cfg["joint_names"])
        qpos_adr, dof_adr, act_ids, rng, tau = [], [], [], [], []

        # Map joint id -> actuator id, checking each actuator drives exactly one joint.
        joint_to_act: dict[int, int] = {}
        for a in range(m.nu):
            if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                raise ValueError(f"actuator {m.actuator(a).name} is not joint-driven")
            j = int(m.actuator_trnid[a, 0])
            if j in joint_to_act:
                raise ValueError(f"joint {m.joint(j).name} has more than one actuator")
            joint_to_act[j] = a

        for n in names:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            if j < 0:
                raise ValueError(f"joint {n!r} not found in model")
            if j not in joint_to_act:
                raise ValueError(f"joint {n!r} has no actuator")
            a = joint_to_act[j]
            if m.actuator_gear[a, 0] != 1.0:
                raise ValueError(f"actuator on {n!r} has gear != 1; torque mapping would be wrong")
            lo, hi = m.actuator_ctrlrange[a]
            if not np.isclose(lo, -hi):
                raise ValueError(f"actuator on {n!r} has asymmetric ctrlrange {lo, hi}")
            qpos_adr.append(m.jnt_qposadr[j])
            dof_adr.append(m.jnt_dofadr[j])
            act_ids.append(a)
            rng.append(m.jnt_range[j])
            tau.append(hi)

        if len(set(act_ids)) != m.nu:
            raise ValueError("model has actuators not covered by the canonical joint list")

        def geom_id(name: str) -> int:
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            if g < 0:
                raise ValueError(f"geom {name!r} not found")
            return g

        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, robot_cfg["imu_site"])
        if site < 0:
            raise ValueError(f"site {robot_cfg['imu_site']!r} not found")

        return cls(
            joint_names=names,
            legs=tuple(robot_cfg["legs"]),
            qpos_adr=np.array(qpos_adr),
            dof_adr=np.array(dof_adr),
            actuator_ids=np.array(act_ids),
            joint_range=np.array(rng),
            torque_limit=np.array(tau),
            foot_geom_ids=np.array([geom_id(g) for g in robot_cfg["foot_geoms"]]),
            floor_geom_id=geom_id(robot_cfg["floor_geom"]),
            imu_site_id=site,
            base_body_id=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"),
        )
