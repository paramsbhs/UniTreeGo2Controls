"""FR1 / Phase 1 acceptance: stand for 30 s, regulate height, respect torque limits."""

import numpy as np
import pytest

from go2_wbc.control.stance_controller import leg_ik_vertical
from go2_wbc.runner import run_stance
from go2_wbc.utils.logging import RunLogger


def test_leg_ik_reproduces_home_pose():
    # The Menagerie 'home' keyframe is thigh 0.9, knee -1.8 at base z = 0.27.
    thigh, knee = leg_ik_vertical(0.27 - 0.022, 0.213, 0.213)
    assert thigh == pytest.approx(0.9, abs=0.06)
    assert knee == pytest.approx(-1.8, abs=0.12)
    assert knee == pytest.approx(-2 * thigh)  # equal link lengths


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stands_for_30_seconds(env_ctrl, seed):
    env, ctrl = env_ctrl
    log = RunLogger(env.idx)
    r = run_stance(env, ctrl, duration=30.0, seed=seed, logger=log)
    assert not r.fell, r.first_fault
    assert r.height_rms_error < 0.005          # 5 mm
    assert r.max_tilt < np.radians(5)
    contacts = np.stack([log.to_table().column(f"contact_{leg}").to_numpy()
                         for leg in env.idx.legs])
    assert contacts[:, -1000:].all(), "a foot lost contact while standing"


def test_height_regulation_tracks_steps(env_ctrl):
    env, ctrl = env_ctrl
    targets = [0.27, 0.22, 0.32]
    sched = lambda t: targets[min(int(t // 4.0), 2)]
    finals = []

    def grab(s):
        if abs(s.time % 4.0 - 3.9) < env.control_dt / 2:  # 0.1 s before each change
            finals.append(s.base_pos[2])

    r = run_stance(env, ctrl, duration=12.0, seed=0, height_schedule=sched, on_step=grab)
    assert not r.fell, r.first_fault
    assert len(finals) == 3
    np.testing.assert_allclose(finals, targets, atol=0.005)


def test_commanded_torque_never_exceeds_limits(env_ctrl):
    env, ctrl = env_ctrl
    log = RunLogger(env.idx)
    run_stance(env, ctrl, duration=3.0, seed=0, logger=log,
               height_schedule=lambda t: 0.18 if t > 0.5 else 0.27)
    table = log.to_table()
    for i, j in enumerate(env.idx.joint_names):
        assert np.all(np.abs(table.column(f"taucmd_{j}").to_numpy())
                      <= env.idx.torque_limit[i] + 1e-9)


def test_push_is_survived(env_ctrl):
    """Not an acceptance test yet: a smoke check that a modest shove doesn't topple it."""
    env, ctrl = env_ctrl

    def shove(s):
        env.set_base_force([40.0, 0.0, 0.0] if 2.0 <= s.time < 2.1 else [0.0, 0.0, 0.0])

    r = run_stance(env, ctrl, duration=5.0, seed=0, on_step=shove)
    assert not r.fell, r.first_fault
