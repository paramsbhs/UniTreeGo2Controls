"""FR1: same seed -> identical initial state and identical trajectory."""

import numpy as np

from go2_wbc.runner import run_stance
from go2_wbc.utils.logging import RunLogger


def test_reset_same_seed_same_state(env):
    a = env.reset(seed=7)
    env.step(np.ones(12))  # disturb the state in between
    b = env.reset(seed=7)
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.base_pos, b.base_pos)
    np.testing.assert_array_equal(a.base_quat, b.base_quat)


def test_reset_different_seed_different_state(env):
    a = env.reset(seed=1)
    b = env.reset(seed=2)
    assert not np.array_equal(a.q, b.q)  # init_noise.joint_pos_std > 0 in the default config


def test_closed_loop_rollout_is_bit_exact(env_ctrl):
    env, ctrl = env_ctrl
    logs = []
    for _ in range(2):
        log = RunLogger(env.idx)
        run_stance(env, ctrl, duration=2.0, seed=3, logger=log)
        logs.append(log.to_table())
    for name in ("base_pz", "q_FL_calf_joint", "taucmd_RR_thigh_joint", "foot_force_RL"):
        a, b = (log.column(name).to_numpy() for log in logs)
        np.testing.assert_array_equal(a, b)


def test_timestep_comes_from_config(cfg):
    from go2_wbc.sim.mujoco_env import Go2Env

    cfg["sim"]["timestep"] = 0.001
    cfg["sim"]["control_decimation"] = 2
    env = Go2Env(cfg)
    assert env.sim_dt == 0.001
    assert env.control_dt == 0.002
