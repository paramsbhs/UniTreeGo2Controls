"""Cross-check the controller-side model (Pinocchio + Unitree URDF) against MuJoCo.

Catches joint-order, quaternion-convention and frame-offset mismatches before they
corrupt the estimator or the QP. See docs/robot_model.md for the known
differences this test documents (2 mm foot offset, ~0.9 kg mass difference).
"""

import mujoco
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from go2_wbc.utils.transforms import rpy_to_quat, wxyz_to_xyzw


@pytest.fixture(scope="module")
def pin_model():
    try:
        from robot_descriptions import go2_description
    except Exception as e:  # network or package missing
        pytest.skip(f"Go2 URDF unavailable: {e}")
    return pin.buildModelFromUrdf(go2_description.URDF_PATH, pin.JointModelFreeFlyer())


def mj_to_pin_q(env, pin_model, base_pos, base_quat_wxyz, q_joints):
    """MuJoCo state -> Pinocchio configuration. Joints are mapped BY NAME."""
    q = np.zeros(pin_model.nq)
    q[:3] = base_pos
    q[3:7] = wxyz_to_xyzw(base_quat_wxyz)
    for i, name in enumerate(env.idx.joint_names):
        q[pin_model.idx_qs[pin_model.getJointId(name)]] = q_joints[i]
    return q


def test_foot_positions_match(env, pin_model):
    data = pin_model.createData()
    rng = np.random.default_rng(0)
    m, d = env.model, env.data
    lo, hi = env.idx.joint_range.T
    worst = 0.0
    for _ in range(25):
        env.reset(seed=0)
        q_j = np.clip(d.qpos[env.idx.qpos_adr] + rng.normal(0, 0.3, 12), lo, hi)
        base_pos = np.array([0.1, -0.2, 0.3])
        base_quat = rpy_to_quat(rng.uniform(-0.4, 0.4, 3))
        d.qpos[:3], d.qpos[3:7] = base_pos, base_quat
        d.qpos[env.idx.qpos_adr] = q_j
        mujoco.mj_kinematics(m, d)

        q_pin = mj_to_pin_q(env, pin_model, base_pos, base_quat, q_j)
        pin.framesForwardKinematics(pin_model, data, q_pin)
        for leg, g in zip(env.idx.legs, env.idx.foot_geom_ids, strict=True):
            p_pin = data.oMf[pin_model.getFrameId(f"{leg}_foot")].translation
            worst = max(worst, np.linalg.norm(p_pin - d.geom_xpos[g]))
    # MJCF foot sphere sits at x=-0.002 in the calf frame; the URDF foot frame at x=0.
    assert worst < 2.5e-3, f"max foot position mismatch {worst * 1000:.2f} mm"


def test_joint_names_exist_in_urdf(env, pin_model):
    for name in env.idx.joint_names:
        assert pin_model.existJointName(name), name


def test_total_mass_close(env, pin_model):
    m_mj = env.model.body_subtreemass[env.idx.base_body_id]
    m_pin = pin.computeTotalMass(pin_model)
    assert abs(m_mj - m_pin) / m_mj < 0.10  # documented mismatch ~6%, see robot_model.md
