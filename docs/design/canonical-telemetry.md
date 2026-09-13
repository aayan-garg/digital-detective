# Canonical Telemetry Representation

This is a design proposal, not an implementation or an implementation-technology decision.

## Goal

Provide a small, preservation-first boundary for a single incident case so future Digital Detective components can consume metrics, logs, and traces without discarding source-specific structure or assuming a universal telemetry schema. The immediate use case is the verified RCAEval Parquet distribution; the boundary must not prevent another dataset from being added later.

## Design Principles

- **Preserve source semantics.** Retain raw fields and record where they came from. Do not reinterpret a field merely because its name resembles a familiar concept.
- **Keep modality-specific structure.** Metrics, logs, and traces remain separate because their schemas, identities, and timestamps differ.
- **Use minimal abstraction.** Standardize only case boundaries, optional modality presence, ground truth separation, and modality-level metadata.
- **Record provenance.** A consumer must be able to identify the source dataset, case, revision, format, and any transformation performed.
- **Do not normalize prematurely.** Do not flatten modalities, canonicalize service names, parse metric names, or infer units until a later verified requirement warrants it.

## Proposed Structure

```text
Case
├── case metadata
├── ground truth metadata
├── metrics (optional)
│   ├── raw data
│   └── schema/provenance metadata
├── logs (optional)
│   ├── raw data
│   └── schema/provenance metadata
└── traces (optional)
    ├── raw data
    └── schema/provenance metadata
```

A case envelope owns the three optional modality containers. Each present modality contains its raw source-shaped data and one shared schema/provenance description. Schema and provenance metadata apply to the whole modality and must not be duplicated on every telemetry row.

This is a structural proposal only. It does not prescribe Python classes, data frames, serialization, storage, or an in-memory representation.

## Case Metadata

For RCAEval, case metadata may preserve only verified case-index concepts:

- case identifier (`case`);
- dataset, suite, system, and system name; and
- incident coverage metadata where present in the index: `inject_time`, time start/end, duration, normal/faulty timestep counts, and modality availability/count fields.

These fields describe the source case; they do not establish a universal incident vocabulary.

## Ground Truth

Ground truth is separate from telemetry and preserves raw source values. For the inspected RCAEval distribution, this includes `root_cause_service`, `fault`, and `inject_time`; `fault_description` and `repetition` may also be preserved when supplied by the case index.

The index observed in discovery has singular `root_cause_service` and `fault` fields. This design does not claim that all RCAEval cases—or future sources—are universally single-label, and it must not force that conclusion.

## Metrics

Metrics initially remain a wide time-series representation with its source fields intact. The inspected RCAEval case has `time` plus numeric metric columns, but metric column sets vary by case and system.

Do not define a universal metric ontology or parse metric names into service and indicator semantics yet. The source timestamp field and its verified representation belong in modality metadata; any unverified unit or semantic interpretation remains unspecified.

## Logs

Preserve raw log rows and their source fields. The inspected RCAEval case establishes `timestamp`, `container_name`, and `message` as observed fields; `timestamp` is verified there as Unix seconds and `container_name` is a service-like identifier.

Do not add normalization rules beyond those verified for the relevant source. In particular, a container name must not automatically be treated as a globally canonical service identity.

## Traces

Preserve raw trace rows and the observed identifiers/timing fields: `time`, `traceID`, `spanID`, `serviceName`, `methodName`, `operationName`, `parentSpanID`, `startTimeMillis`, `startTime`, `duration`, and `statusCode`.

Trace identifiers stay as source strings. The exact units and semantics of the inspected `startTime` and `duration` fields are unresolved, so this representation must retain them without assigning inferred meanings. `startTimeMillis` is also retained as an original field rather than used to silently overwrite another trace time field.

## Provenance

Each present modality has one minimal provenance record containing:

- source dataset;
- source case;
- source revision or version when available;
- original format;
- original field names;
- timestamp field;
- identity field or fields; and
- transformation status.

For example, the RCAEval Parquet distribution can identify `metrics.parquet`, `logs.parquet`, or `traces.parquet` as the original format/file context and record their source field names. Transformation status must distinguish raw preservation from any future transformation; it must not imply that a transformation is required now.

## Explicit Non-Goals

This milestone does **not** include:

- anomaly detection;
- RCA or graph construction;
- RAG or agents;
- deep learning or remediation;
- a universal telemetry ontology;
- metric semantic parsing;
- service-name canonicalization;
- inferred timestamp or duration semantics; or
- persistence, database, or serialization design.

It also does not introduce dataclasses, Pydantic, pandas, PyArrow, database layers, serialization frameworks, plugin systems, or generic schema engines.

## Acceptance Criteria

An eventual implementation satisfies this design when:

1. One case can represent metrics-only data.
2. One case can represent metrics, logs, and traces together.
3. Every modality is independently optional.
4. Raw source fields are preservable without requiring conversion into a shared flat row schema.
5. Modality-level provenance is preservable once per modality.
6. Ground truth remains distinct from telemetry and preserves supplied raw RCAEval values.
7. No unverified timestamp, unit, duration, identity, or metric-name interpretation is required to create a case.
8. The representation does not force all modalities into one flat schema.
9. A later dataset can be represented by supplying its own raw modality data and provenance without redesigning the case envelope.
10. The model remains small enough to understand without a framework or generic schema engine.

## Test Cases

Future implementation tests should cover:

- an RE1-style metrics-only case, with logs and traces absent;
- an RE2-style multi-source case containing metrics, logs, and traces;
- a case whose logs are absent;
- a case whose traces are absent;
- preservation of raw metric, log, and trace field names;
- preservation of source ground-truth fields such as `root_cause_service`, `fault`, and `inject_time` outside telemetry;
- preservation of modality-level provenance, including source revision when provided; and
- rejection or avoidance of unsupported semantic assumptions: no required parsing of metric names, canonicalization of service-like identifiers, or inferred trace time/duration units.

These are test descriptions only; no tests are added by this design note.

## Open Questions

- The precise units and semantics of RCAEval trace `startTime` and `duration` remain unverified.
- The original RCAEval JSON/CSV archive schemas have not been directly inspected; only the current official Parquet distribution was inspected.
- The inspected RCAEval index has one service and one fault field per case, but it does not establish whether broader RCAEval usage or future sources require multiple ground-truth entities.
- A universal semantic mapping from metric suffixes to units or fault indicators has not been verified.
