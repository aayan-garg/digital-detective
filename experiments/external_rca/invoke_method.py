"""Isolated subprocess entry point for RCAEval e2e functions."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcaeval-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--method", choices=("baro", "rcd", "causalrca", "microrank", "mmrcd"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.rcaeval_root))
    from RCAEval.e2e import baro, causalrca, microrank, mmrcd, rcd

    case_dir = args.dataset_root / args.case_id
    index = pd.read_parquet(args.dataset_root / "cases.parquet")
    rows = index[index["case"] == args.case_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one case-index row for {args.case_id!r}, found {len(rows)}")
    row = rows.iloc[0]
    inject_time = int((case_dir / "inject_time.txt").read_text(encoding="utf-8").strip())
    if inject_time != int(row["inject_time"]):
        raise ValueError(f"inject_time mismatch for {args.case_id}")

    metrics = pd.read_parquet(case_dir / "metrics.parquet").replace([float("inf"), float("-inf")], pd.NA)
    metrics = metrics.ffill().fillna(0)
    metrics = metrics.loc[:, ~metrics.columns.str.endswith("_latency-50")]
    metrics = metrics.rename(
        columns={c: c.replace("_latency-90", "_latency") for c in metrics.columns if c.endswith("_latency-90")}
    )
    normal = metrics[metrics["time"] < inject_time].tail(600)
    anomalous = metrics[metrics["time"] >= inject_time].head(600)
    metric_window = pd.concat([normal, anomalous], ignore_index=True)
    dataset = "re2-ob"

    method_output = io.StringIO()
    with redirect_stdout(method_output):
        if args.method == "microrank":
            traces_path = case_dir / "traces.parquet"
            if not traces_path.is_file():
                raise FileNotFoundError("MicroRank requires traces.parquet")
            traces = pd.read_parquet(traces_path)
            # RE2-OB Parquet traces use microseconds, while the benchmark index
            # and inject_time.txt use Unix seconds. The published MicroRank
            # function compares these fields directly and expects the
            # trace-unit timestamp.
            trace_inject_time = inject_time * 1_000_000
            result = microrank(traces, inject_time=trace_inject_time, dataset=dataset)
        elif args.method == "mmrcd":
            required = ("logts.csv", "tracets_err.csv", "tracets_lat.csv")
            missing = [name for name in required if not (case_dir / name).is_file()]
            if missing:
                raise RuntimeError(
                    "RCAEval Multi-source RCD requires published pre-aggregated inputs; "
                    f"missing {', '.join(missing)}"
                )
            result = mmrcd(
                {
                    "metric": metric_window,
                    "logts": pd.read_csv(case_dir / "logts.csv"),
                    "tracets_err": pd.read_csv(case_dir / "tracets_err.csv"),
                    "tracets_lat": pd.read_csv(case_dir / "tracets_lat.csv"),
                },
                inject_time=inject_time,
                dataset=dataset,
            )
        else:
            function = {"baro": baro, "rcd": rcd, "causalrca": causalrca}[args.method]
            result = function(metric_window, inject_time=inject_time, dataset=dataset)

    print(
        json.dumps(
            {
                "ranks": result["ranks"],
                "ground_truth": str(row["root_cause_service"]),
                "inject_time": inject_time,
                "telemetry_window": {
                    "mode": "rcaeval_oracle",
                    "normal_start": int(normal["time"].iloc[0]),
                    "normal_end": int(normal["time"].iloc[-1]),
                    "incident_start": int(anomalous["time"].iloc[0]),
                    "incident_end": int(anomalous["time"].iloc[-1]),
                },
                "compatibility_patch_applied": args.method == "microrank",
                "patch_description": (
                    "Converted benchmark injection seconds to trace microseconds "
                    "at the experiment adapter boundary."
                    if args.method == "microrank"
                    else None
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
