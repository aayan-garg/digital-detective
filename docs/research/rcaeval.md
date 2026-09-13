# RCAEval Discovery

Discovery date: 2026-09-14. This note records inspected source facts only; it does not define a Digital Detective internal schema or adapter.

## Sources

- **Repository:** [phamquiluan/RCAEval](https://github.com/phamquiluan/RCAEval), inspected at live `main` commit [`bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90`](https://github.com/phamquiluan/RCAEval/tree/bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90). The repository declares an MIT license. Its README instructs users to clone the repository, create a Python 3.12 environment, and install `.[default]`; the `1.2.0` tag resolves to `5f22afb1cd9e383f52c41c2e8e99c8ef930db5d8`, which is not the inspected live commit.
- **Dataset:** [phamquiluan/RCAEval on Hugging Face](https://huggingface.co/datasets/phamquiluan/RCAEval), inspected on its `main` revision. The file listing showed short commit `afeacb1`; the inspected `re2ob_checkoutservice_cpu_1` directory manifest provided per-file object IDs and sizes. The dataset card declares MIT and links this repository and the paper.
- **Paper:** [RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data](https://arxiv.org/abs/2412.17015), arXiv:2412.17015. Its abstract describes three datasets, 735 failure cases, three microservice systems, and an evaluation framework with 15 baselines. The current repository/dataset are the source for the file- and schema-level facts below.

## Verified Dataset Structure

The current Hugging Face `cases.parquet` index has 735 rows and these fields:

`case`, `dataset`, `suite`, `system`, `system_name`, `root_cause_service`, `fault`, `fault_description`, `repetition`, `inject_time`, `n_metrics`, `n_timesteps`, `time_start`, `time_end`, `duration_minutes`, `normal_timesteps`, `faulty_timesteps`, `has_logs`, `n_logs`, `has_traces`, `n_traces`, and `has_root_cause_file`.

The repository README describes nine datasets in three suites—RE1, RE2, and RE3—across Online Boutique (OB), Sock Shop (SS), and Train Ticket (TT), totaling 735 cases:

- RE1: 375 metric-only cases; CPU, memory, disk, delay, and loss faults.
- RE2: 270 multi-source cases; RE1 fault types plus socket faults. SS has no traces; OB and TT do.
- RE3: 90 multi-source, code-level-fault cases (F1–F5); SS has no traces.

Each current case is a directory named `{suite}{system}_{service}_{fault}_{repetition}`. The current Parquet layout uses `metrics.parquet`, `inject_time.txt`, and, where present, `logs.parquet` and `traces.parquet`. The repository documents older/original counterparts as `metrics.json`, `logs.csv`, and `traces.csv`; this discovery did not download those original archives.

The case index is the authoritative selection mechanism inspected here: it stores the single `root_cause_service`, `fault`, and `inject_time` for each row, as well as per-signal availability and size counts. The index contains singular service and fault fields; no multi-label ground-truth field was observed. That is not proof that every possible RCA question has only one causal entity.

## One Real Case

**Verified case:** `re2ob_checkoutservice_cpu_1` (RE2-OB, Online Boutique).

The directly read index row identifies `root_cause_service = "checkoutservice"`, `fault = "cpu"`, `fault_description = "CPU stress"`, `repetition = 1`, and `inject_time = 1705354566`. It reports 72 metric columns, 1,441 metric timesteps, 171,322 log rows, 391,997 trace rows, `has_logs = true`, `has_traces = true`, and `has_root_cause_file = false`. The directory manifest contained:

- `metrics.parquet` — 157,212 bytes;
- `logs.parquet` — 336,876 bytes;
- `traces.parquet` — 10,146,857 bytes; and
- `inject_time.txt` — 10 bytes.

The selected case is therefore about 10.6 MB of telemetry, plus the case index. This is a directly inspected multi-source case, not a claim about the size of every case.

### Metrics

`metrics.parquet` has 1,441 rows. `time` is an `int64` Unix-seconds column (first value `1705353846`); the other 72 fields are `double` metrics. Names use a component prefix and underscore-delimited signal family, for example `checkoutservice_cpu`, `checkoutservice_mem`, `checkoutservice_socket`, `checkoutservice_workload`, and `checkoutservice_latency-50`/`checkoutservice_latency-90`. Observed components include `adservice`, `cartservice`, `checkoutservice`, `currencyservice`, `emailservice`, `frontend`, `paymentservice`, `productcatalogservice`, `recommendationservice`, `redis`, and `shippingservice`; `frontend-external` also occurs. Observed signal families include `cpu`, `mem`, `diskio` (only some components), `socket`, `workload`, `error`, `latency-50`, and `latency-90`.

### Logs

`logs.parquet` has 171,322 rows and exactly these fields: `timestamp: int64`, `container_name: large_string`, and `message: large_string`. The first inspected row was timestamp `1705353846`, container `currencyservice`, message `conversion request successful`. Thus the log timestamp in this case is Unix seconds; the service-like identifier is `container_name`.

### Traces

`traces.parquet` has 391,997 rows and exactly these fields: `time: large_string`, `traceID: large_string`, `spanID: large_string`, `serviceName: large_string`, `methodName: large_string`, `operationName: large_string`, `parentSpanID: large_string`, `startTimeMillis: int64`, `startTime: int64`, `duration: int64`, and `statusCode: int64`. The first row had `time = "21:24"`, `serviceName = "currencyservice"`, a string trace ID/span ID, `startTimeMillis = 1705353846065`, `startTime = 1705353846065999`, `duration = 186`, and `statusCode = 0`. The exact units/semantics of `startTime` and `duration` were not verified from source documentation and must not be assumed.

## Ground Truth

For the inspected case, ground truth is unambiguously represented in `cases.parquet` by `root_cause_service = "checkoutservice"` and `fault = "cpu"`; the same injection epoch is stored as `inject_time` and in `inject_time.txt`. The case directory itself has no `root_cause.txt` (`has_root_cause_file = false`).

The live evaluator uses a ranked sequence under the `"ranks"` key. For metric-style rankings, it derives service predictions from the string portion before the first underscore and evaluates a fine-grained `Node(service, indicator)` answer as well as a service-only answer. Its current fault-to-indicator mapping uses `cpu`, `mem`, `socket`, and (for delay/loss) `latency`; disk uses a dataset-specific `disk_metric` variable. This evaluator mapping is implementation behavior, not a replacement for preserving the raw dataset label.

## Evaluation

The paper abstract states that RCAEval supports coarse- and fine-grained RCA; the inspected live evaluator makes this concrete for rank-based outputs:

- Baseline entry points return an ordered list of strings in `{"ranks": [...]}`.
- For each case, the evaluator computes accuracy at ranks 1 through 5. Its `accuracy(k)` is the fraction of cases whose answer appears in the first `k` ranked items.
- `Avg@k` is the mean of `AC@1` through `AC@k`. The command-line benchmark prints service-level `Avg@5` by fault category; the generated evaluation data also records top-1, top-3, top-5, and Avg@5 for service and metric levels.
- The current evaluator truncates stored rankings to five items. With `--report-chance`, it can compute chance and lift only when the candidate count is recorded for every evaluated case.

The official runner is `python main.py --dataset DATASET --method METHOD`; the README lists baseline implementations in `RCAEval.e2e`. No baseline was installed or run during this discovery.

## Download/Access Method

The current README recommends the Hugging Face Parquet copy and documents `snapshot_download` with `allow_patterns`, allowing a suite or selected case paths to be retrieved rather than the full dataset. The inspected index (`cases.parquet`, 29,500 bytes) was downloaded first to select a case. The four files for `re2ob_checkoutservice_cpu_1` were then retrieved directly from the official Hugging Face paths and read with temporary, non-project PyArrow tooling. No benchmark suite, source dataset archive, or telemetry artifact was added to this repository.

## Important Implications for Digital Detective

- A future adapter must not assume one fixed metric column set or one fixed modality set: availability differs by suite/system/case, and the index exposes that availability explicitly.
- The raw timestamp fields have different names and representations across metrics, logs, and traces. Preserve source field names and unit provenance until their semantics are verified; do not collapse them based on naming alone.
- The inspected benchmark ground truth supports at least service-level and fault/indicator-level rank evaluation, but the live evaluator contains compatibility mappings. Future evaluation integration must preserve raw `root_cause_service`, `fault`, and `inject_time` before applying any benchmark-specific mapping.
- These are evidence-based constraints only. They are not a decision about Digital Detective's canonical telemetry schema.

## Unknowns

- The precise units and semantics of trace `startTime` and `duration` were not verified.
- The original JSON/CSV archive schemas were not directly inspected; only the current official Parquet distribution was read.
- The inspected index stores one service and one fault per case. Whether any broader interpretation of RCAEval should allow multiple simultaneous ground-truth entities is not established by this discovery.
- No source was inspected to establish a universal semantic mapping from every metric suffix to a unit or fault indicator. Such mappings must be verified per dataset/version before implementation.
