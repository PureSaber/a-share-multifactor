"""Rebuild requirements.lock from scratch with the certified Python 3.10 toolchain."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "requirements.lock"
DISPLAY_COMMAND = "python tools/rebuild_lock.py"
PIP_COMPILE_ARGUMENTS = (
    "--extra",
    "dev",
    "--build-deps-for",
    "editable",
    "--allow-unsafe",
    "--strip-extras",
    "--resolver",
    "backtracking",
    "--newline",
    "lf",
    "--index-url",
    "https://pypi.org/simple",
    "--find-links",
    "vendor/wheels",
    "--constraint",
    "constraints/runtime.txt",
    "--output-file",
    "requirements.lock",
    "pyproject.toml",
)


def main() -> int:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "requirements.lock must be rebuilt with Python 3.10; "
            f"received {sys.version_info.major}.{sys.version_info.minor}"
        )

    previous = OUTPUT.read_bytes() if OUTPUT.exists() else None
    if OUTPUT.exists():
        OUTPUT.unlink()
    environment = os.environ.copy()
    environment["CUSTOM_COMPILE_COMMAND"] = DISPLAY_COMMAND
    command = (sys.executable, "-m", "piptools", "compile", *PIP_COMPILE_ARGUMENTS)
    try:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    except BaseException:
        if previous is not None:
            OUTPUT.write_bytes(previous)
        raise

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"requirements.lock sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
