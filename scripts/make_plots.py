"""Plot a run log: body height, orientation, joint torques, foot contact.

    uv run python scripts/make_plots.py                 # newest log in results/logs/
    uv run python scripts/make_plots.py path/to/run.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from go2_wbc import REPO_ROOT
from go2_wbc.utils.logging import load_run

LEGS = ("FL", "FR", "RL", "RR")


def newest_log() -> Path:
    logs = sorted((REPO_ROOT / "results" / "logs").glob("*.parquet"))
    if not logs:
        raise SystemExit("no logs in results/logs/ - run a script with --log first")
    return logs[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", type=Path)
    args = ap.parse_args()
    path = args.log or newest_log()
    table, meta = load_run(path)
    c = {name: table.column(name).to_numpy() for name in table.column_names}
    t = c["t"]

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True, constrained_layout=True)

    ax[0].plot(t, c["base_pz"] * 100, label="measured")
    ax[0].plot(t, c["h_ref"] * 100, "--", label="setpoint")
    ax[0].set_ylabel("base height [cm]")
    ax[0].legend(loc="best")

    for k in ("roll", "pitch"):
        ax[1].plot(t, np.degrees(c[k]), label=k)
    ax[1].set_ylabel("orientation [deg]")
    ax[1].legend(loc="best")

    for leg in LEGS:
        for j in ("thigh", "calf"):
            ax[2].plot(t, c[f"tau_{leg}_{j}_joint"], lw=0.8, label=f"{leg} {j}")
    ax[2].set_ylabel("joint torque [N m]")
    ax[2].legend(loc="upper right", ncol=4, fontsize=7)

    for leg in LEGS:
        ax[3].plot(t, c[f"foot_force_{leg}"], lw=0.8, label=leg)
    ax[3].set_ylabel("foot normal force [N]")
    ax[3].set_xlabel("time [s]")
    ax[3].legend(loc="upper right", ncol=4)

    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{path.name}  (seed={meta.get('summary', {}).get('seed')})")

    out = REPO_ROOT / "results" / "figures" / f"{path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"figure: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
