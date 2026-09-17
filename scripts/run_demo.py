#!/usr/bin/env python3
"""Run the Digital Detective investigation demo (deterministic or agentic mode)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Detective Demo Runner")
    parser.add_argument(
        "--mode",
        choices=["deterministic", "agent", "both"],
        default="deterministic",
        help="Investigation mode: 'deterministic' (engine only), 'agent' (Ollama/Mock agent), or 'both'",
    )
    args, unknown = parser.parse_known_args()

    repo_root = Path(__file__).resolve().parent.parent
    if args.mode == "deterministic":
        cmd = [sys.executable, "-m", "digital_detective.detective.demo"] + unknown
    elif args.mode == "agent":
        cmd = [sys.executable, "-m", "digital_detective.agent.demo", "--mode", "agent"] + unknown
    else:
        cmd = [sys.executable, "-m", "digital_detective.agent.demo", "--mode", "both"] + unknown

    print(f"Running demo: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=repo_root)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
