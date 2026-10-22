"""YAML configuration loading.

Configs are plain nested dicts. `load_config` resolves paths relative to the repo
root, so scripts and tests work from any working directory.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from go2_wbc import REPO_ROOT


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve(path)) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(
    robot: str | Path = "configs/robot/go2.yaml",
    sim: str | Path = "configs/sim/default.yaml",
    controller: str | Path = "configs/controller/stance.yaml",
    overrides: dict | None = None,
) -> dict[str, Any]:
    """Load the three config groups into one dict: {'robot', 'sim', 'controller'}."""
    cfg = {
        "robot": load_yaml(robot),
        "sim": load_yaml(sim),
        "controller": load_yaml(controller),
    }
    return deep_merge(cfg, overrides or {})
