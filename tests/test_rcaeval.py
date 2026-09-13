from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from digital_detective.rcaeval import load_rcaeval_case


class RCAEvalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_metrics_only_case(self) -> None:
        self._write_case("re1ob_adservice_cpu_1")

        case = load_rcaeval_case(self.root, "re1ob_adservice_cpu_1")

        self.assertIsNotNone(case.metrics)
        self.assertIsNone(case.logs)
        self.assertIsNone(case.traces)
        self.assertEqual(case.metrics.raw_data.column_names, ["time", "adservice_cpu"])
        self.assertEqual(case.metrics.provenance.original_format, "metrics.parquet")

    def test_loads_multisource_case_with_raw_fields_and_provenance(self) -> None:
        self._write_case("re2ob_checkoutservice_cpu_1", logs=True, traces=True)

        case = load_rcaeval_case(self.root, "re2ob_checkoutservice_cpu_1", "dataset-revision")

        self.assertEqual(case.logs.raw_data.column_names, ["timestamp", "container_name", "message"])
        self.assertEqual(
            case.traces.raw_data.column_names,
            ["time", "traceID", "spanID", "serviceName", "duration"],
        )
        self.assertEqual(case.traces.raw_data.to_pylist()[0]["duration"], 123)
        self.assertEqual(case.logs.provenance.source_dataset, "RE2-OB")
        self.assertEqual(case.logs.provenance.source_case, "re2ob_checkoutservice_cpu_1")
        self.assertEqual(case.logs.provenance.source_revision, "dataset-revision")
        self.assertEqual(case.metrics.provenance.original_format, "metrics.parquet")
        self.assertEqual(case.logs.provenance.original_format, "logs.parquet")
        self.assertEqual(case.traces.provenance.original_format, "traces.parquet")
        self.assertEqual(case.logs.provenance.timestamp_field, "timestamp")
        self.assertEqual(case.logs.provenance.identity_fields, ("container_name",))
        self.assertEqual(case.traces.provenance.identity_fields, ("traceID", "spanID", "serviceName"))

    def test_optional_modalities_are_independent(self) -> None:
        self._write_case("with_logs", logs=True)
        self._write_case("with_traces", traces=True)

        with_logs = load_rcaeval_case(self.root, "with_logs")
        with_traces = load_rcaeval_case(self.root, "with_traces")

        self.assertIsNotNone(with_logs.logs)
        self.assertIsNone(with_logs.traces)
        self.assertIsNone(with_traces.logs)
        self.assertIsNotNone(with_traces.traces)

    def test_keeps_ground_truth_separate_from_telemetry(self) -> None:
        self._write_case("ground_truth")

        case = load_rcaeval_case(self.root, "ground_truth")

        self.assertEqual(case.ground_truth.values["root_cause_service"], "checkoutservice")
        self.assertEqual(case.ground_truth.values["fault"], "cpu")
        self.assertEqual(case.ground_truth.values["inject_time"], 1705354566)
        self.assertNotIn("root_cause_service", case.metrics.raw_data.column_names)
        self.assertEqual(case.metadata.incident_metadata["has_logs"], False)

    def test_requires_exact_case_lookup(self) -> None:
        self._write_case("duplicate")
        self._append_index_row("duplicate")

        with self.assertRaisesRegex(ValueError, "occurs 2 times"):
            load_rcaeval_case(self.root, "duplicate")
        with self.assertRaisesRegex(ValueError, "was not found"):
            load_rcaeval_case(self.root, "missing")

    def test_rejects_missing_or_corrupt_required_inputs(self) -> None:
        self._write_case("missing_metrics")
        (self.root / "missing_metrics" / "metrics.parquet").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "metrics"):
            load_rcaeval_case(self.root, "missing_metrics")

        self._write_case("malformed_inject", inject_text="not-a-time")
        with self.assertRaisesRegex(ValueError, "Malformed.*injection"):
            load_rcaeval_case(self.root, "malformed_inject")

        self._write_case("decimal_inject", inject_text="1705354566.0")
        with self.assertRaisesRegex(ValueError, "Malformed.*injection"):
            load_rcaeval_case(self.root, "decimal_inject")

        self._write_case("mismatched_inject", inject_text="1705354567")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_rcaeval_case(self.root, "mismatched_inject")

    def test_rejects_corrupt_optional_modality_when_present(self) -> None:
        self._write_case("corrupt_logs", logs=True)
        (self.root / "corrupt_logs" / "logs.parquet").write_text("not parquet", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Could not read RCAEval logs"):
            load_rcaeval_case(self.root, "corrupt_logs")

    def _write_case(
        self,
        case_id: str,
        *,
        logs: bool = False,
        traces: bool = False,
        inject_text: str = "1705354566",
    ) -> None:
        self._append_index_row(case_id)
        case_directory = self.root / case_id
        case_directory.mkdir()
        pq.write_table(
            pa.table({"time": [1705354566], "adservice_cpu": [0.75]}),
            case_directory / "metrics.parquet",
        )
        (case_directory / "inject_time.txt").write_text(inject_text, encoding="utf-8")
        if logs:
            pq.write_table(
                pa.table(
                    {
                        "timestamp": [1705354566],
                        "container_name": ["checkoutservice"],
                        "message": ["raw log message"],
                    }
                ),
                case_directory / "logs.parquet",
            )
        if traces:
            pq.write_table(
                pa.table(
                    {
                        "time": ["raw-time"],
                        "traceID": ["trace-1"],
                        "spanID": ["span-1"],
                        "serviceName": ["checkoutservice"],
                        "duration": [123],
                    }
                ),
                case_directory / "traces.parquet",
            )

    def _append_index_row(self, case_id: str) -> None:
        index_path = self.root / "cases.parquet"
        rows = pq.read_table(index_path).to_pylist() if index_path.exists() else []
        rows.append(
            {
                "case": case_id,
                "dataset": "RE2-OB",
                "suite": "RE2",
                "system": "ob",
                "system_name": "Online Boutique",
                "root_cause_service": "checkoutservice",
                "fault": "cpu",
                "fault_description": "CPU stress",
                "repetition": 1,
                "inject_time": 1705354566,
                "n_metrics": 2,
                "n_timesteps": 1,
                "time_start": 1705354500,
                "time_end": 1705354600,
                "duration_minutes": 1,
                "normal_timesteps": 0,
                "faulty_timesteps": 1,
                "has_logs": False,
                "n_logs": 0,
                "has_traces": False,
                "n_traces": 0,
                "has_root_cause_file": False,
            }
        )
        pq.write_table(pa.Table.from_pylist(rows), index_path)


if __name__ == "__main__":
    unittest.main()
