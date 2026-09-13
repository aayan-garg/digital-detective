# RCAEval Adapter

This is a design proposal for one RCAEval-specific adapter. It does not implement loading, choose a dependency, or alter the canonical telemetry model.

## Scope

The future adapter reads one locally available case from the current official RCAEval Parquet distribution and returns one `TelemetryCase`. It is responsible for RCAEval file names, case-index fields, and modality availability. It is not a generic dataset framework and does not implement analysis, evaluation, normalization, persistence, or remote download.

The smallest proposed public operation is conceptually:

```text
load_rcaeval_case(dataset_root, case_id, source_revision=None) -> TelemetryCase
```

`dataset_root` contains `cases.parquet` and case directories; `case_id` is selected from the index's `case` field. The implementation must use the selected index row rather than infer ground truth from the case-directory name.

## Actual Verified RCAEval Input Layout

This design was re-verified against the public Hugging Face dataset `phamquiluan/RCAEval` at `main` revision `afeacb11bcc94dadfd1c8f483ee4377b2b8b614e` and the source repository at `bb48c5aa9a24f1d5fcc716bdd479ea2d63145c90`.

### Case selection

`cases.parquet` is the verified index. It has one row per case and includes `case`, `dataset`, `suite`, `system`, `system_name`, `root_cause_service`, `fault`, `fault_description`, `repetition`, `inject_time`, metric coverage fields, and modality availability/count fields. A caller supplies an exact `case` value; the adapter must find exactly one matching index row or fail clearly.

### Local layout

Each case is a directory named like `re1ob_adservice_cpu_1` or `re2ob_checkoutservice_cpu_1`. The verified current files are:

```text
dataset_root/
├── cases.parquet
└── {case_id}/
    ├── metrics.parquet       # required
    ├── inject_time.txt       # required
    ├── logs.parquet          # optional
    └── traces.parquet        # optional
```

The inspected RE1 case `re1ob_adservice_cpu_1` contains only `metrics.parquet` and `inject_time.txt`. The inspected RE2 case `re2ob_checkoutservice_cpu_1` contains all four files. Logs and traces are therefore optional per case; their availability must be determined from the actual case files, with the index availability fields retained as source metadata rather than treated as a substitute for file checks.

### Verified schemas and values relevant to mapping

- Metrics are a wide Parquet table. The inspected RE2 case has `time: int64` and varying numeric metric columns. Column names and sets differ by case/system.
- Logs, when present, are a Parquet table. The inspected RE2 case fields are `timestamp: int64`, `container_name: large_string`, and `message: large_string`.
- Traces, when present, are a Parquet table. The inspected RE2 case fields are `time`, `traceID`, `spanID`, `serviceName`, `methodName`, `operationName`, `parentSpanID`, `startTimeMillis`, `startTime`, `duration`, and `statusCode`; trace identifiers are strings.
- `inject_time.txt` holds a single decimal Unix timestamp in the inspected cases. The matching case-index value is `inject_time`.

The original JSON/CSV archive layout is outside this adapter's scope.

## Mapping from RCAEval Input to `TelemetryCase`

| RCAEval source | `TelemetryCase` destination | Treatment |
| --- | --- | --- |
| index `case` | `metadata.case_id` | Preserve exactly. |
| index `dataset`, `suite`, `system`, `system_name` | matching `CaseMetadata` fields | Preserve exactly. |
| other index coverage/availability fields | `metadata.incident_metadata` | Preserve raw field/value pairs; do not calculate replacements. |
| index `root_cause_service`, `fault`, `fault_description`, `repetition`, `inject_time` | `ground_truth.values` | Preserve separately from telemetry. |
| `metrics.parquet` | `metrics.raw_data` | Preserve the table read from source without reshaping or parsing names. |
| `logs.parquet`, when present | `logs.raw_data` | Preserve the table read from source without parsing messages or canonicalizing containers. |
| `traces.parquet`, when present | `traces.raw_data` | Preserve the table read from source without parsing operations or reinterpreting timing. |
| `inject_time.txt` | validation only | Parse as an integer and require equality with index `inject_time`; do not add a second normalized representation. |

The future implementation may use one direct Parquet reader dependency because Parquet cannot be decoded by the Python standard library. Dependency selection is deferred; it must be verified against the project Python range and current package API before implementation. No dataframe library or generic dataset framework is required by this design.

## What Remains Raw and Preserved

- The original table object and all source columns for each present modality.
- Exact metric column names, including source prefixes/suffixes.
- Log message strings and `container_name` values.
- Trace IDs, span IDs, service/operation fields, and all timing/duration fields.
- The index's original ground-truth values and case metadata values.
- The source revision supplied to the adapter, when available.

The adapter does not convert wide metrics to long form, aggregate telemetry, fill missing values, decode log content, derive dependencies, or add aliases.

## Metadata and Provenance

For each present modality, create one `ModalityProvenance` with:

- `source_dataset`: the index `dataset` value;
- `source_case`: the index `case` value;
- `source_revision`: the explicit adapter argument, or `None` when unavailable;
- `original_format`: the source filename (`metrics.parquet`, `logs.parquet`, or `traces.parquet`);
- `original_field_names`: the fields read from that table, in source order;
- `timestamp_field`: `time` for metrics, `timestamp` for logs, and `time` for traces in the currently inspected distribution; and
- `identity_fields`: empty for metrics; `container_name` for logs; and `traceID`, `spanID`, and `serviceName` for traces.

Set `transformation_status` to `"raw"`. These field declarations record source layout; they do not assert semantic equivalence among timestamp or identity values.

## Missing Optional Modalities and Errors

- `metrics.parquet`, `inject_time.txt`, and a unique matching index row are required. Missing, unreadable, malformed, or inconsistent required inputs must raise a clear adapter-specific error that identifies the case and failed input.
- `logs.parquet` and `traces.parquet` are optional. If absent, set the corresponding `TelemetryCase` modality to `None` and do not create provenance for it.
- If an optional file exists but cannot be read as its expected source table, fail clearly rather than silently treating corruption as absence.
- If `inject_time.txt` cannot be parsed as one integer or differs from the index's `inject_time`, fail clearly. The adapter must not select one value silently.
- If a required table lacks the source timestamp field described above, fail clearly. It must not guess an alternate field.

## What the Adapter Must Not Infer

The adapter must not:

- infer trace timestamp units or duration semantics;
- parse metrics into service/indicator semantics;
- canonicalize service, container, trace, or operation names;
- assume that all cases have logs or traces;
- derive or alter ground truth from the directory name;
- turn the RCAEval singular index fields into a universal multi-label model;
- fill, aggregate, resample, sort, or otherwise normalize raw modality data; or
- add evaluation, anomaly-detection, RCA, graph, RAG, agent, or remediation behavior.

## Acceptance Criteria

Future implementation is complete when it:

1. Loads one real RE1 metrics-only case from a local current Parquet distribution and returns metrics with `logs is None` and `traces is None`.
2. Loads one real RE2 multi-source case and returns metrics, logs, and traces.
3. Selects the case by exact lookup in `cases.parquet`, requiring exactly one row.
4. Preserves raw telemetry fields and source-shaped table data without flattening or semantic parsing.
5. Populates `CaseMetadata`, separate `GroundTruth`, and per-modality `ModalityProvenance` from the verified source fields.
6. Preserves `root_cause_service`, `fault`, and `inject_time` in ground truth without deriving replacements.
7. Represents missing optional files as absent modalities, but rejects unreadable optional files rather than hiding corruption.
8. Rejects missing/corrupt required files, an invalid index row, an unparseable injection time, and an index/file injection-time mismatch with useful case-specific errors.
9. Requires no timestamp-unit, duration, service-name, or metric-semantic normalization.
10. Adds only the smallest verified Parquet-reading dependency required for implementation, if one is needed.

## Concrete Tests to Implement Later

- Use a small local fixture shaped as `re1ob_adservice_cpu_1`: index row, `metrics.parquet`, and `inject_time.txt`; assert a metrics-only case and raw metric fields.
- Use a small local fixture shaped as `re2ob_checkoutservice_cpu_1`: index row plus all four files; assert all three modalities, raw log/trace fields, and modality-level provenance.
- Verify a case with no `logs.parquet` yields `logs is None`; separately verify missing `traces.parquet` yields `traces is None`.
- Verify `root_cause_service`, `fault`, and `inject_time` are present only in `GroundTruth.values`, not injected into raw telemetry or parsed from `case_id`.
- Verify field names are retained exactly, including `checkoutservice_latency-90`, `container_name`, `traceID`, and `startTime`.
- Verify provenance field names, source dataset/case/revision, original filename, timestamp field declaration, identity fields, and `"raw"` transformation status.
- Verify failures for a missing `metrics.parquet`, malformed `inject_time.txt`, injection-time mismatch, no matching index row, multiple matching index rows, and corrupt optional Parquet.
- Verify no adapter output changes a trace `duration`, converts a trace ID, parses a metric name, or canonicalizes a service-like identifier.

The fixture format and dependency test strategy are intentionally deferred until a Parquet reader is selected and verified.

## Open Questions / Limitations

- The exact semantics/units of trace `startTime` and `duration` remain unverified; this adapter records fields but does not interpret them.
- The design covers the current official Parquet distribution only, not the older JSON/CSV archives.
- The current case index stores singular service and fault fields; this adapter preserves them without making a universal claim about ground-truth cardinality.
- The per-case index data was verified in discovery and re-verified for selected manifests, but a future implementation must inspect actual local files again at the chosen dataset revision before pinning a Parquet reader API.
- Remote download, caching, selective synchronisation, and dataset-version resolution are deliberately out of scope; the adapter reads a local dataset root.
