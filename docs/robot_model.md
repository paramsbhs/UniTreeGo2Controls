# Robot model and conventions

Everything here is enforced by a test somewhere in `tests/`. If you change a
convention, change the test first.

## Sources

| Model | Used by | Source |
|---|---|---|
| MJCF `models/go2/go2.xml` | Simulator (the "plant") | MuJoCo Menagerie, pinned in `models/go2/SOURCE.md` |
| URDF `go2_description.urdf` | Controller-side model (Pinocchio) | Unitree `unitree_ros`, fetched by `robot_descriptions` |

The controller deliberately uses a *different* model from the simulator. That is
what a real robot looks like: your model never matches the plant exactly.
Known differences are listed below; the cross-check lives in `tests/test_pinocchio_fk.py`.

## Frames

- **World:** z up, gravity −z, origin on the floor.
- **Base (body):** MuJoCo body `base`, x forward, y left, z up. Origin is the
  free-joint origin (not the COM; the COM is at +21 mm x, −5 mm z).
- **IMU:** site `imu`, at (−25.6, 0, 42.3) mm in the base frame, axes aligned with base.
- **Feet:** collision spheres `FL`, `FR`, `RL`, `RR` (radius 22 mm). "Foot position"
  means the sphere centre, not the contact point.

## State conventions

| Quantity | Convention |
|---|---|
| Quaternion | **[w, x, y, z]** everywhere. Pinocchio uses [x, y, z, w]: convert with `wxyz_to_xyzw` only at the Pinocchio boundary. |
| `SimState.base_lin_vel` | **world** frame (MuJoCo free-joint `qvel[0:3]`). Use `base_lin_vel_body` for body frame. |
| `SimState.base_ang_vel` | **body** frame (MuJoCo free-joint `qvel[3:6]`). |
| Pinocchio free-flyer `v` | linear **and** angular in body (local) frame. Note the mismatch with MuJoCo's linear velocity. |
| Euler angles | ZYX roll/pitch/yaw, logging and plots only. Never as internal state. |

## Joint ordering

Canonical order (defined in `configs/robot/go2.yaml`, never inferred from the MJCF):

| idx | joint | type | range [rad] | torque limit [N·m] |
|---|---|---|---|---|
| 0 | FL_hip_joint | hip (abduction, x-axis) | [−1.047, 1.047] | 23.7 |
| 1 | FL_thigh_joint | thigh (pitch, y-axis) | [−1.571, 3.491] | 23.7 |
| 2 | FL_calf_joint | calf (knee, y-axis) | [−2.723, −0.838] | 45.43 |
| 3–5 | FR_* | same | same | same |
| 6–8 | RL_* | thigh range [−0.524, 4.538] | | same |
| 9–11 | RR_* | thigh range [−0.524, 4.538] | | same |

`RobotIndex` maps this order onto `qpos`/`qvel`/`ctrl` addresses **by name**, and
fails at load time if a joint is missing, has two actuators, or has gear ≠ 1.

Sign conventions: positive thigh swings the foot backwards; the knee is always
negative (bent). Nominal stance (`home` keyframe): hip 0, thigh 0.9, knee −1.8,
base height 0.27 m.

## Actuators

Menagerie `go2.xml` uses `<motor>` actuators: `ctrl` **is joint torque** (gear 1,
no internal servo), clamped by MuJoCo to `ctrlrange`. Do not use `go2_mjx.xml`;
its actuators are position servos and would silently add a PD under yours.

Not modelled (yet): motor torque–speed curve, gear-ratio reflected inertia beyond
the `armature=0.01` already in the MJCF, control latency, encoder quantisation.
These are the realism additions planned for Phase 5.

The MJCF also has **joint damping 2 N·m·s/rad and friction loss 0.2 N·m** on every
joint. Your controller doesn't know about them, so they show up as unmodelled
dissipation. Remember this when identifying gains.

## Sensors

The Menagerie model ships without sensors. `Go2Env` adds at load time (via `MjSpec`,
without editing the vendored XML):

- `imu_gyro`: angular velocity, IMU frame, rad/s
- `imu_acc`: specific force (includes +g when stationary), IMU frame, m/s²

Both are noiseless. Noise and bias models belong to the estimator phase.

## Contact

A foot is "in contact" when its total normal force against the floor exceeds
`contact.force_threshold` (5 N). This is **ground truth** from MuJoCo's contact
solver. The Phase 2 estimator must replace it with something a real robot could
measure, and the experiments should report both.

## Known model mismatches (MJCF vs URDF)

| Item | MJCF | URDF | Impact |
|---|---|---|---|
| Total mass | 15.21 kg | 16.09 kg | ~6 %: the stance feedforward uses the MJCF value. The WBC should use the URDF and show it is robust to this. |
| Calf mass | 0.241 kg | 0.194 kg (incl. foot) | small |
| Foot point | sphere at x = −2 mm in calf frame | frame at x = 0 | ≤ 2 mm foot position error (asserted < 2.5 mm) |

## Leg geometry (used by the stance IK)

Thigh 0.213 m (hip-pitch axis → knee axis), calf 0.213 m (knee → foot centre).
Equal lengths mean that with the foot directly under the hip, `knee = −2 · thigh`.
