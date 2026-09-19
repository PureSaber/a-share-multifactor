"""Overlay the jointly reviewed sibling checkouts on an installed locked environment.

Run with the Python interpreter of the intended virtual environment. This does
not resolve new dependency versions or alter the frozen release lock.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    siblings = [root.parent / "quant-data-kit", root.parent / "quant-lab", root]
    for repo in siblings:
        if not (repo / "pyproject.toml").is_file():
            raise FileNotFoundError(f"Missing reviewed sibling repository: {repo}")
    uv = shutil.which("uv")
    command = [uv, "pip", "install", "--python", sys.executable, "--no-deps"] if uv else [
        sys.executable, "-m", "pip", "install", "--no-deps"
    ]
    for repo in siblings:
        command.extend(["--editable", str(repo)])
    subprocess.run(command, check=True, cwd=root)
    print("Installed local QDK, quant-lab and A-share changes; release lock remains unchanged.")


if __name__ == "__main__":
    main()
