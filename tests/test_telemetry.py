"""Tests for the preservation-first telemetry representation."""

import unittest

from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


def provenance(
    *,
    original_format: str,
    fields: tuple[str, ...],
    timestamp_field: str | None,
    identity_fields: tuple[str, ...] = (),
) -> ModalityProvenance:
    return ModalityProvenance(
        source_dataset="RCAEval",
        source_case="re2ob_checkoutservice_cpu_1",
        source_revision="afeacb1",
        original_format=original_format,
        original_field_names=fields,
        timestamp_field=timestamp_field,
        identity_fields=identity_fields,
    )


class TelemetryCaseTests(unittest.TestCase):
    def test_metrics_only_case_keeps_other_modalities_absent(self) -> None:
        metrics = [{"time": 1705353846, "checkoutservice_cpu": 0.2}]
        case = TelemetryCase(
            metadata=CaseMetadata(
                case_id="re1ob_checkoutservice_cpu_1",
                dataset="RE1-OB",
                suite="RE1",
                system="ob",
                system_name="Online Boutique",
            ),
            metrics=TelemetryModality(
                raw_data=metrics,
                provenance=provenance(
                    original_format="metrics.parquet",
                    fields=("time", "checkoutservice_cpu"),
                    timestamp_field="time",
                ),
            ),
        )

        self.assertIs(case.metrics.raw_data, metrics)
        self.assertIsNone(case.logs)
        self.assertIsNone(case.traces)

    def test_multisource_case_preserves_raw_data_and_provenance(self) -> None:
        metrics = [{"time": 1705353846, "checkoutservice_latency-90": 0.85}]
        logs = [{"timestamp": 1705353846, "container_name": "currencyservice", "message": "ok"}]
        traces = [
            {
                "time": "21:24",
                "traceID": "00123",
                "spanID": "abc",
                "serviceName": "currencyservice",
                "startTime": 1705353846065999,
                "duration": 186,
            }
        ]
        case = TelemetryCase(
            metadata=CaseMetadata(
                case_id="re2ob_checkoutservice_cpu_1",
                dataset="RE2-OB",
                suite="RE2",
                system="ob",
                system_name="Online Boutique",
                incident_metadata={"n_metrics": 72, "n_logs": 171322, "n_traces": 391997},
            ),
            ground_truth=GroundTruth(
                values={
                    "root_cause_service": "checkoutservice",
                    "fault": "cpu",
                    "inject_time": 1705354566,
                }
            ),
            metrics=TelemetryModality(
                raw_data=metrics,
                provenance=provenance(
                    original_format="metrics.parquet",
                    fields=("time", "checkoutservice_latency-90"),
                    timestamp_field="time",
                ),
            ),
            logs=TelemetryModality(
                raw_data=logs,
                provenance=provenance(
                    original_format="logs.parquet",
                    fields=("timestamp", "container_name", "message"),
                    timestamp_field="timestamp",
                    identity_fields=("container_name",),
                ),
            ),
            traces=TelemetryModality(
                raw_data=traces,
                provenance=provenance(
                    original_format="traces.parquet",
                    fields=("time", "traceID", "spanID", "serviceName", "startTime", "duration"),
                    timestamp_field="time",
                    identity_fields=("traceID", "spanID", "serviceName"),
                ),
            ),
        )

        self.assertIs(case.metrics.raw_data, metrics)
        self.assertIs(case.logs.raw_data, logs)
        self.assertIs(case.traces.raw_data, traces)
        self.assertEqual(case.traces.raw_data[0]["traceID"], "00123")
        self.assertEqual(case.logs.provenance.original_field_names, ("timestamp", "container_name", "message"))
        self.assertEqual(case.traces.provenance.identity_fields, ("traceID", "spanID", "serviceName"))
        self.assertEqual(case.metrics.provenance.transformation_status, "raw")
        self.assertEqual(case.ground_truth.values["root_cause_service"], "checkoutservice")
        self.assertNotIn("root_cause_service", case.metadata.incident_metadata)

    def test_each_modality_can_be_absent_independently(self) -> None:
        metadata = CaseMetadata(case_id="another-source-case")

        no_logs = TelemetryCase(metadata=metadata, traces=TelemetryModality(
            raw_data=[{"trace_id": "1"}],
            provenance=provenance(
                original_format="source-traces",
                fields=("trace_id",),
                timestamp_field=None,
                identity_fields=("trace_id",),
            ),
        ))
        no_traces = TelemetryCase(metadata=metadata, logs=TelemetryModality(
            raw_data=[{"event_time": "unverified", "text": "message"}],
            provenance=provenance(
                original_format="source-logs",
                fields=("event_time", "text"),
                timestamp_field="event_time",
            ),
        ))

        self.assertIsNone(no_logs.metrics)
        self.assertIsNone(no_logs.logs)
        self.assertIsNotNone(no_logs.traces)
        self.assertIsNone(no_traces.metrics)
        self.assertIsNotNone(no_traces.logs)
        self.assertIsNone(no_traces.traces)

    def test_unknown_semantics_are_not_required(self) -> None:
        raw_trace = {"duration": 186, "startTime": 1705353846065999}
        modality = TelemetryModality(
            raw_data=[raw_trace],
            provenance=provenance(
                original_format="traces.parquet",
                fields=("startTime", "duration"),
                timestamp_field=None,
            ),
        )

        self.assertIs(modality.raw_data[0], raw_trace)
        self.assertIsNone(modality.provenance.timestamp_field)
        self.assertEqual(modality.raw_data[0]["duration"], 186)
