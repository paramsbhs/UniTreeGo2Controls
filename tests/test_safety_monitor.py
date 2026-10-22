import numpy as np
import pytest

from go2_wbc.control.safety_monitor import SafetyMonitor
from go2_wbc.utils.transforms import rpy_to_quat


@pytest.fixture
def mon(env, cfg):
    return SafetyMonitor.from_config(env.idx, cfg["controller"])


def test_nominal_state_has_no_faults(env, mon):
    s = env.reset(seed=0)
    assert mon.check(s, np.zeros(12)) == []
    assert not mon.tripped


def test_nan_torque_trips(env, mon):
    s = env.reset(seed=0)
    tau = np.zeros(12)
    tau[3] = np.nan
    faults = mon.check(s, tau)
    assert [f.kind for f in faults] == ["nonfinite_torque"]
    assert mon.tripped


def test_excessive_tilt_trips(env, mon):
    s = env.reset(seed=0)
    s.base_quat = rpy_to_quat([0.8, 0.0, 0.0])
    assert "tilt" in [f.kind for f in mon.check(s, np.zeros(12))]
    assert mon.tripped


def test_trip_is_latched(env, mon):
    s = env.reset(seed=0)
    mon.check(s, np.full(12, np.nan))
    mon.check(s, np.zeros(12))
    assert mon.tripped


def test_over_limit_torque_is_a_warning_not_a_trip(env, mon):
    s = env.reset(seed=0)
    faults = mon.check(s, np.full(12, 100.0))
    assert [f.kind for f in faults] == ["torque_limit"]
    assert not mon.tripped


def test_env_refuses_nonfinite_torque(env):
    env.reset(seed=0)
    with pytest.raises(FloatingPointError):
        env.step(np.full(12, np.inf))
