#!/usr/bin/env python3
"""Run the complete Digital Detective test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "pytest", "tests/"] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)} in {repo_root}")
    res = subprocess.run(cmd, cwd=repo_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
