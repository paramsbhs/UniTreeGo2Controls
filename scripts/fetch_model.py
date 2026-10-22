"""Re-fetch the Go2 MJCF from MuJoCo Menagerie into models/go2/.

The model is vendored in the repo (pinned commit in models/go2/SOURCE.md), so you
only need this to update it.

    uv run python scripts/fetch_model.py [--ref <commit-or-branch>]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from go2_wbc import REPO_ROOT

URL = "https://github.com/google-deepmind/mujoco_menagerie.git"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    args = ap.parse_args()

    dest = REPO_ROOT / "models" / "go2"
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "menagerie"
        git = ["git", "-C", str(repo)]
        subprocess.run(["git", "clone", "--filter=blob:none", "--sparse", URL, str(repo)],
                       check=True)
        subprocess.run([*git, "sparse-checkout", "set", "unitree_go2"], check=True)
        subprocess.run([*git, "checkout", args.ref], check=True)
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(repo / "unitree_go2", dest,
                        ignore=shutil.ignore_patterns("*mjx*", "*.png"))
    (dest / "SOURCE.md").write_text(
        f"Vendored from {URL}\nPath: unitree_go2/\nCommit: {sha}\n"
        "License: BSD-3-Clause (see LICENSE)\n"
    )
    print(f"models/go2 updated to menagerie@{sha[:10]} - rerun pytest")


if __name__ == "__main__":
    main()
