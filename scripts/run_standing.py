"""Phase 1 demo: the Go2 stands and regulates its body height.

Headless (fast, for experiments):
    uv run python scripts/run_standing.py --duration 30 --log
With the viewer (macOS needs mjpython):
    uv run mjpython scripts/run_standing.py --viewer
Height-step test:
    uv run python scripts/run_standing.py --height-steps 0.27 0.22 0.32 --step-every 4 --log
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from go2_wbc import REPO_ROOT
from go2_wbc.config import load_config
from go2_wbc.runner import build, run_stance
from go2_wbc.utils.logging import RunLogger


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer (real time)")
    ap.add_argument("--height-steps", type=float, nargs="+", default=None,
                    help="sequence of body-height setpoints (m)")
    ap.add_argument("--step-every", type=float, default=4.0, help="seconds per height setpoint")
    ap.add_argument("--log", action="store_true", help="write a Parquet log to results/logs/")
    ap.add_argument("--controller-config", default="configs/controller/stance.yaml")
    ap.add_argument("--sim-config", default="configs/sim/default.yaml")
    args = ap.parse_args()

    cfg = load_config(sim=args.sim_config, controller=args.controller_config)
    env, ctrl = build(cfg)

    schedule = None
    if args.height_steps:
        steps, every = args.height_steps, args.step_every
        schedule = lambda t: steps[min(int(t // every), len(steps) - 1)]

    logger = RunLogger(env.idx) if args.log else None
    kwargs = dict(duration=args.duration, seed=args.seed, height_schedule=schedule, logger=logger)

    if args.viewer:
        import mujoco.viewer

        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as v:
                summary = run_stance(env, ctrl, viewer=v, realtime=True, **kwargs)
        except RuntimeError as e:
            if sys.platform == "darwin" and "mjpython" in str(e):
                print("On macOS the viewer needs mjpython:\n"
                      "    uv run mjpython scripts/run_standing.py --viewer", file=sys.stderr)
                return 2
            raise
    else:
        summary = run_stance(env, ctrl, **kwargs)

    print(summary.report())

    if logger is not None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = logger.save(
            REPO_ROOT / "results" / "logs" / f"standing-{stamp}.parquet",
            meta={"script": "run_standing", "args": vars(args), "config": cfg,
                  "summary": summary.__dict__},
        )
        print(f"log: {path.relative_to(REPO_ROOT)}")
    return 1 if summary.fell else 0


if __name__ == "__main__":
    raise SystemExit(main())
