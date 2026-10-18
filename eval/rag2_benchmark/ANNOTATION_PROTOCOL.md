# DD-IR-60 Human Relevance Annotation Protocol

## Overview

The DD-IR-60 benchmark evaluates retrieval of operational runbooks for microservice incidents. While previous automated evaluations relied on **operational category-grounded proxy labels** (e.g., whether a retrieved runbook addresses the relevant fault category), genuine retrieval evaluation requires expert human judgment.

This document describes the annotation protocol for grading the candidate pool in `eval/rag2_benchmark/human_annotation_stratified_template.tsv`.

## Grading Scale (0 – 2)

Each candidate query-document pair `(query_id, document_id)` must be independently graded on a 3-point relevance scale:

| Grade | Label | Description | Operational Criteria |
|:---:|:---|:---|:---|
| **0** | **Not Relevant** | The document does not discuss the fault, the affected component, or relevant mitigation. | Generic OpenTelemetry specifications, architectural overview diagrams, or runbooks for completely unrelated systems/faults. |
| **1** | **Partially Relevant** | The document describes related background, telemetry concepts, or diagnostic tooling, but does not provide actionable remediation steps. | E.g., an article explaining how Kubernetes measures CPU limits or how gRPC handles connection timeouts, without specific commands. |
| **2** | **Highly Relevant / Actionable** | The document contains direct diagnostic commands, mitigation steps, or rollback procedures relevant to resolving the incident. | E.g., Kubernetes runbook for CPU throttling explaining `kubectl top`, vertical pod autoscaling, or adjusting CPU limits/requests. |

## Annotation Guidelines

1. **Independent Evaluation**: Annotators must evaluate the document based strictly on the incident symptoms and affected service described in the query.
2. **Actionability Over Vocabulary**: A document that shares keywords with the query but provides no operational guidance should be graded `0`.
3. **Safety Verification**: Ensure the runbook contains safety warnings, preconditions, or rollback guidance before assigning grade `2`.

## Current Benchmark Status

- Automated proxy benchmark results are recorded in `eval/rag_evaluation_results.json`.
- Category agreement serves as a necessary heuristic baseline, NOT proof of manual semantic correctness.
- The stratified pool contains 36 candidate pairs across 12 representative incidents spanning 6 fault families (`cpu`, `delay`, `disk`, `loss`, `mem`, `socket`).
