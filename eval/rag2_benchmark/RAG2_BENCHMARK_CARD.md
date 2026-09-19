# DD-IR-60 benchmark card

## Corpus

The final corpus is a separate, frozen operational-knowledge snapshot built from pinned official Kubernetes, Prometheus, gRPC, and OpenTelemetry repository documentation. It is temporally leakage-controlled with respect to benchmark information availability. No RE2 case, fault label, postmortem, RCA report, annotation, or remediation result was used to select or write a unit.

- Retrieval units: 536
- Source families: 4
- Source versions: 4
- Excluded source units: 47
- Categories: cpu, debugging, dependency-failure, disk, dns, error-rate, grpc, logging, memory, metrics, networking, observability, process, resource-limits, safe-operations, service-health, tracing
- Human qrels: not present
- Final C0-C4 evaluation: not run

Repository licenses were independently verified from the pinned repositories: Kubernetes website CC BY 4.0; Prometheus, gRPC, and OpenTelemetry Apache-2.0.
