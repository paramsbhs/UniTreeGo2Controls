"""Run logging to Parquet.

One row per control step, flat named columns (e.g. q_FL_calf_joint, foot_force_RR),
plus run metadata (config, seed, versions) stored in the Parquet schema metadata,
so every log file is self-describing and reproducible.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from go2_wbc.robot import RobotIndex
from go2_wbc.sim.mujoco_env import SimState
from go2_wbc.utils.transforms import quat_to_rpy


class RunLogger:
    def __init__(self, idx: RobotIndex):
        self.idx = idx
        self.rows: list[dict[str, float]] = []

    def log(self, s: SimState, tau_cmd: np.ndarray, **extra: float) -> None:
        row: dict[str, float] = {"t": s.time}
        for k, axis in enumerate("xyz"):
            row[f"base_p{axis}"] = s.base_pos[k]
            row[f"base_v{axis}"] = s.base_lin_vel[k]
            row[f"base_w{axis}"] = s.base_ang_vel[k]
            row[f"imu_gyro_{axis}"] = s.imu_gyro[k]
            row[f"imu_acc_{axis}"] = s.imu_acc[k]
        for k, c in enumerate("wxyz"):
            row[f"base_q{c}"] = s.base_quat[k]
        for name, v in zip(("roll", "pitch", "yaw"), quat_to_rpy(s.base_quat), strict=True):
            row[name] = v
        for i, j in enumerate(self.idx.joint_names):
            row[f"q_{j}"] = s.q[i]
            row[f"dq_{j}"] = s.dq[i]
            row[f"tau_{j}"] = s.tau[i]
            row[f"taucmd_{j}"] = tau_cmd[i]
        for k, leg in enumerate(self.idx.legs):
            row[f"foot_force_{leg}"] = s.foot_force[k]
            row[f"contact_{leg}"] = float(s.foot_contact[k])
            for a, axis in enumerate("xyz"):
                row[f"foot_{leg}_{axis}"] = s.foot_pos[k, a]
        row.update({k: float(v) for k, v in extra.items()})
        self.rows.append(row)

    def to_table(self, meta: dict | None = None) -> pa.Table:
        cols = {k: [r.get(k, np.nan) for r in self.rows] for k in self.rows[0]}
        table = pa.table(cols)
        info = {
            "mujoco_version": mujoco.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            **(meta or {}),
        }
        return table.replace_schema_metadata({"go2_wbc": json.dumps(info, default=str)})

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.to_table(meta), path)
        return path


def load_run(path: str | Path):
    """Return (pandas.DataFrame-like table, metadata dict)."""
    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata.get(b"go2_wbc", b"{}"))
    return table, meta
