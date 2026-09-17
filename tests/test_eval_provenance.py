"""Unit tests for BenchmarkProvenance and report provenance tracking.
"""

import json
import unittest

from eval.harness import BenchmarkExecutionReport
from eval.models import (
    AggregateMethodMetrics,
    BenchmarkProvenance,
)


class TestBenchmarkProvenance(unittest.TestCase):
    def test_provenance_deterministic_serialization(self) -> None:
        prov1 = BenchmarkProvenance(
            evaluator_schema_version="1.0.0",
            manifest_id="re2_ob_rep1_smoke_v1",
            manifest_hash="abc123def456",
            window_mode="oracle",
            candidate_universe_policy="canonical_v1",
            dataset_name="RE2-OB",
            methods=("s_comb", "trace_elevation"),
            dataset_artifact_version="zenodo_v2",
            dataset_source_doi="10.5281/zenodo.123456",
            repository_revision="260157a",
            experiment_config_id="configs/eval_smoke_re2_ob.json",
            modality_policy="all_available",
            random_seed=42,
            runtime_info={"python_version": "3.12.0", "platform": "Windows-10"},
        )
        prov2 = BenchmarkProvenance(
            evaluator_schema_version="1.0.0",
            manifest_id="re2_ob_rep1_smoke_v1",
            manifest_hash="abc123def456",
            window_mode="oracle",
            candidate_universe_policy="canonical_v1",
            dataset_name="RE2-OB",
            methods=("s_comb", "trace_elevation"),
            dataset_artifact_version="zenodo_v2",
            dataset_source_doi="10.5281/zenodo.123456",
            repository_revision="260157a",
            experiment_config_id="configs/eval_smoke_re2_ob.json",
            modality_policy="all_available",
            random_seed=42,
            runtime_info={"python_version": "3.12.0", "platform": "Windows-10"},
        )

        d1 = prov1.to_dict()
        d2 = prov2.to_dict()
        self.assertEqual(d1, d2)
        s1 = json.dumps(d1, sort_keys=True)
        s2 = json.dumps(d2, sort_keys=True)
        self.assertEqual(s1, s2)

    def test_missing_optional_fields_explicit_unspecified(self) -> None:
        prov = BenchmarkProvenance(
            evaluator_schema_version="1.0.0",
            manifest_id="re2_ob_rep1_smoke_v1",
            manifest_hash="abc123def456",
            window_mode="detected",
            candidate_universe_policy="canonical_v1",
            dataset_name="RE2-OB",
            methods=("s_comb",),
            dataset_artifact_version=None,
            dataset_source_doi=None,
            repository_revision=None,
            experiment_config_id=None,
        )
        d = prov.to_dict()
        self.assertEqual(d["dataset_artifact_version"], "UNSPECIFIED")
        self.assertEqual(d["dataset_source_doi"], "UNSPECIFIED")
        self.assertEqual(d["repository_revision"], "UNSPECIFIED")
        self.assertEqual(d["experiment_config_id"], "UNSPECIFIED")

        # from_dict roundtrip handles UNSPECIFIED
        restored = BenchmarkProvenance.from_dict(d)
        self.assertIsNone(restored.dataset_artifact_version)
        self.assertIsNone(restored.dataset_source_doi)
        self.assertIsNone(restored.repository_revision)
        self.assertIsNone(restored.experiment_config_id)

    def test_report_includes_provenance(self) -> None:
        prov = BenchmarkProvenance(
            evaluator_schema_version="1.0.0",
            manifest_id="smoke_v1",
            manifest_hash="hash123",
            window_mode="oracle",
            candidate_universe_policy="canonical_v1",
            dataset_name="RE2-OB",
            methods=("s_comb",),
        )
        report = BenchmarkExecutionReport(
            schema_version="1.0.0",
            manifest_id="smoke_v1",
            manifest_description="Smoke test",
            window_mode="oracle",
            total_cases=1,
            case_ids=("case_1",),
            evaluated_methods=("s_comb",),
            method_aggregates={},
            headroom_analysis=None,
            case_results=[],
            provenance=prov,
        )
        rep_dict = report.to_dict()
        self.assertIn("provenance", rep_dict)
        self.assertEqual(rep_dict["provenance"]["manifest_hash"], "hash123")
        self.assertEqual(rep_dict["provenance"]["candidate_universe_policy"], "canonical_v1")

    def test_backward_compatibility_without_provenance(self) -> None:
        report = BenchmarkExecutionReport(
            schema_version="1.0.0",
            manifest_id="legacy_v1",
            manifest_description="Legacy report",
            window_mode="oracle",
            total_cases=1,
            case_ids=("case_1",),
            evaluated_methods=("s_comb",),
            method_aggregates={},
            headroom_analysis=None,
            case_results=[],
            provenance=None,
        )
        rep_dict = report.to_dict()
        self.assertNotIn("provenance", rep_dict)


if __name__ == "__main__":
    unittest.main()
