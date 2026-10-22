"""The canonical joint order must map onto the MJCF by name, never by position."""

import mujoco
import numpy as np
import pytest

from go2_wbc.robot import RobotIndex


def test_every_canonical_joint_maps_to_its_own_actuator(env):
    m, idx = env.model, env.idx
    assert idx.n_joints == 12 == m.nu
    assert len(set(idx.actuator_ids)) == 12
    for i, name in enumerate(idx.joint_names):
        joint_id = m.actuator_trnid[idx.actuator_ids[i], 0]
        assert m.joint(joint_id).name == name
        assert m.jnt_qposadr[joint_id] == idx.qpos_adr[i]
        assert m.jnt_dofadr[joint_id] == idx.dof_adr[i]


def test_actuators_are_torque_motors_with_expected_limits(env):
    m, idx = env.model, env.idx
    for i in range(idx.n_joints):
        a = idx.actuator_ids[i]
        assert m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
        assert m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE  # pure torque, no servo
        expected = 45.43 if idx.joint_type(i) == "calf" else 23.7
        assert idx.torque_limit[i] == pytest.approx(expected)


def test_torque_is_applied_to_the_named_joint(env):
    """Push one canonical joint and check only that DOF receives actuator force."""
    env.reset(seed=0)
    for i in range(env.idx.n_joints):
        tau = np.zeros(12)
        tau[i] = 5.0
        env.data.ctrl[env.idx.actuator_ids] = tau
        mujoco.mj_forward(env.model, env.data)
        qfrc = env.data.qfrc_actuator[env.idx.dof_adr]
        assert qfrc[i] == pytest.approx(5.0)
        assert np.count_nonzero(np.abs(qfrc) > 1e-9) == 1


def test_reordered_config_is_respected(cfg, env):
    """Reversing the configured order must reverse the index map, not break it."""
    rcfg = dict(cfg["robot"], joint_names=list(reversed(cfg["robot"]["joint_names"])))
    rev = RobotIndex.from_model(env.model, rcfg)
    np.testing.assert_array_equal(rev.qpos_adr, env.idx.qpos_adr[::-1])


def test_unknown_joint_fails_loudly(cfg, env):
    rcfg = dict(cfg["robot"], joint_names=["FL_hip_joint", "not_a_joint"])
    with pytest.raises(ValueError, match="not_a_joint"):
        RobotIndex.from_model(env.model, rcfg)
