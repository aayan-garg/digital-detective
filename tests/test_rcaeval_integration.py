"""Integration tests for RCAEval adapter against real dataset files when available.

These tests are skipped by default unless RCAEval_DATASET_ROOT is set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path
import unittest

from digital_detective.rcaeval import load_rcaeval_case

DATASET_ROOT_ENV = "RCAEval_DATASET_ROOT"
DATASET_REVISION_ENV = "RCAEval_DATASET_REVISION"


class RCAEvalRealDataIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset_root = os.environ.get(DATASET_ROOT_ENV)
        if not dataset_root:
            raise unittest.SkipTest(
                f"{DATASET_ROOT_ENV} environment variable not set; skipping real-data integration tests"
            )
        cls.root = Path(dataset_root)
        if not cls.root.is_dir() or not (cls.root / "cases.parquet").is_file():
            raise unittest.SkipTest(
                f"{DATASET_ROOT_ENV}={dataset_root} does not contain cases.parquet"
            )
        cls.revision = os.environ.get(DATASET_REVISION_ENV)

    def test_re1_metrics_only_real_case(self) -> None:
        case_id = "re1ob_adservice_cpu_1"
        if not (self.root / case_id / "metrics.parquet").is_file():
            self.skipTest(f"Case {case_id} not present in {self.root}")

        case = load_rcaeval_case(self.root, case_id, source_revision=self.revision)

        # Metadata
        self.assertEqual(case.metadata.case_id, case_id)
        self.assertEqual(case.metadata.dataset, "RE1-OB")
        self.assertEqual(case.metadata.suite, "RE1")
        self.assertEqual(case.metadata.system, "ob")
        self.assertEqual(case.metadata.system_name, "Online Boutique")

        # Modalities presence/absence
        self.assertIsNotNone(case.metrics)
        self.assertIsNone(case.logs)
        self.assertIsNone(case.traces)

        # Raw field names preserved
        metric_columns = case.metrics.raw_data.column_names
        self.assertIn("time", metric_columns)
        self.assertIn("adservice_cpu", metric_columns)
        self.assertEqual(
            len([col for col in metric_columns if col != "time"]),
            case.metadata.incident_metadata["n_metrics"],
        )

        # Ground truth
        self.assertEqual(case.ground_truth.values["root_cause_service"], "adservice")
        self.assertEqual(case.ground_truth.values["fault"], "cpu")
        self.assertEqual(case.ground_truth.values["inject_time"], 1685202688)
        self.assertEqual(case.ground_truth.values["repetition"], 1)
        self.assertEqual(case.ground_truth.values["fault_description"], "CPU stress")

        # Provenance
        prov = case.metrics.provenance
        self.assertEqual(prov.source_dataset, "RE1-OB")
        self.assertEqual(prov.source_case, case_id)
        self.assertEqual(prov.source_revision, self.revision)
        self.assertEqual(prov.original_format, "metrics.parquet")
        self.assertEqual(prov.original_field_names, tuple(metric_columns))
        self.assertEqual(prov.timestamp_field, "time")
        self.assertEqual(prov.identity_fields, ())
        self.assertEqual(prov.transformation_status, "raw")

        # Row counts match index
        self.assertEqual(
            case.metrics.raw_data.num_rows,
            case.metadata.incident_metadata["n_timesteps"],
        )

    def test_re2_multisource_real_case(self) -> None:
        case_id = "re2ob_checkoutservice_cpu_1"
        if not (self.root / case_id / "metrics.parquet").is_file():
            self.skipTest(f"Case {case_id} not present in {self.root}")

        case = load_rcaeval_case(self.root, case_id, source_revision=self.revision)

        # Metadata
        self.assertEqual(case.metadata.case_id, case_id)
        self.assertEqual(case.metadata.dataset, "RE2-OB")
        self.assertEqual(case.metadata.suite, "RE2")
        self.assertEqual(case.metadata.system, "ob")
        self.assertEqual(case.metadata.system_name, "Online Boutique")

        # All three modalities present
        self.assertIsNotNone(case.metrics)
        self.assertIsNotNone(case.logs)
        self.assertIsNotNone(case.traces)

        # Raw field names preserved without semantic normalization
        metric_columns = case.metrics.raw_data.column_names
        self.assertIn("time", metric_columns)
        self.assertIn("checkoutservice_cpu", metric_columns)
        self.assertIn("checkoutservice_latency-90", metric_columns)

        self.assertEqual(
            case.logs.raw_data.column_names,
            ["timestamp", "container_name", "message"],
        )
        self.assertEqual(
            case.traces.raw_data.column_names,
            [
                "time",
                "traceID",
                "spanID",
                "serviceName",
                "methodName",
                "operationName",
                "parentSpanID",
                "startTimeMillis",
                "startTime",
                "duration",
                "statusCode",
            ],
        )

        # Ground truth
        self.assertEqual(case.ground_truth.values["root_cause_service"], "checkoutservice")
        self.assertEqual(case.ground_truth.values["fault"], "cpu")
        self.assertEqual(case.ground_truth.values["inject_time"], 1705354566)
        self.assertEqual(case.ground_truth.values["repetition"], 1)
        self.assertEqual(case.ground_truth.values["fault_description"], "CPU stress")

        # Provenance for all three modalities
        self.assertEqual(case.metrics.provenance.original_format, "metrics.parquet")
        self.assertEqual(case.metrics.provenance.timestamp_field, "time")
        self.assertEqual(case.metrics.provenance.identity_fields, ())

        self.assertEqual(case.logs.provenance.original_format, "logs.parquet")
        self.assertEqual(case.logs.provenance.timestamp_field, "timestamp")
        self.assertEqual(case.logs.provenance.identity_fields, ("container_name",))

        self.assertEqual(case.traces.provenance.original_format, "traces.parquet")
        self.assertEqual(case.traces.provenance.timestamp_field, "time")
        self.assertEqual(
            case.traces.provenance.identity_fields,
            ("traceID", "spanID", "serviceName"),
        )

        # Comparison with case index counts
        self.assertEqual(
            case.metrics.raw_data.num_rows,
            case.metadata.incident_metadata["n_timesteps"],
        )
        self.assertEqual(
            case.logs.raw_data.num_rows,
            case.metadata.incident_metadata["n_logs"],
        )
        self.assertEqual(
            case.traces.raw_data.num_rows,
            case.metadata.incident_metadata["n_traces"],
        )


if __name__ == "__main__":
    unittest.main()
