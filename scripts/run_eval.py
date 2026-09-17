#!/usr/bin/env python3
"""Run the frozen RCAEval benchmark smoke evaluation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "eval.harness",
        "--smoke-re2-ob",
        "--window-mode",
        "oracle",
    ] + sys.argv[1:]
    print(f"Running evaluation: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=repo_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
