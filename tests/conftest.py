import pytest

from go2_wbc.config import load_config
from go2_wbc.runner import build


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def env_ctrl(cfg):
    return build(cfg)


@pytest.fixture
def env(env_ctrl):
    return env_ctrl[0]
