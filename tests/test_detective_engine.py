"""End-to-end integration tests for InvestigationEngine.
"""

import os
from pathlib import Path
import unittest

from digital_detective.detective.investigator import InvestigationEngine
from digital_detective.rcaeval import load_rcaeval_case


def _get_case_path() -> Path | None:
    dataset_root = os.environ.get("RCAEVAL_DATASET_ROOT")
    if not dataset_root:
        dataset_root = str(Path.home() / ".cache" / "rcaeval_validation")
    root = Path(dataset_root)
    case_dir = root / "re2ob_checkoutservice_cpu_1"
    if case_dir.is_dir():
        return root
    # Also check repository root / cases / re2_ob if available
    repo_case = Path(__file__).resolve().parent.parent / "cases" / "re2_ob"
    if (repo_case / "re2ob_checkoutservice_cpu_1").is_dir():
        return repo_case
    return None


class TestDetectiveEngine(unittest.TestCase):
    def setUp(self) -> None:
        root = _get_case_path()
        if root is None:
            raise unittest.SkipTest("re2ob_checkoutservice_cpu_1 dataset not found; skipping integration test")
        self.case = load_rcaeval_case(root, "re2ob_checkoutservice_cpu_1")

    def test_investigation_end_to_end_with_override(self) -> None:
        engine = InvestigationEngine(budget=20)
        state = engine.investigate(
            self.case,
            window_mode="oracle",
            allow_low_confidence_remediation=True,
        )

        # 1. State status
        self.assertEqual(state.status, "REMEDIATION_AUTHORIZED")
        self.assertTrue(state.remediation_authorized)

        # 2. Budget adherence
        self.assertLessEqual(state.total_query_cost, 20)
        self.assertGreaterEqual(state.remaining_budget, 0)
        self.assertEqual(state.total_query_cost + state.remaining_budget, 20)

        # 3. Candidate universe preservation
        self.assertGreater(len(state.candidate_universe), 5)
        self.assertEqual(len(state.current_rankings), len(state.candidate_universe))

        # 4. Root cause localization
        self.assertIsNotNone(state.decision)
        self.assertEqual(state.decision.root_cause_service, "checkoutservice")
        self.assertGreater(state.decision.confidence, 0.4)

        # 5. Causal consistency validation
        cc = state.decision.causal_consistency
        self.assertIsNotNone(cc)
        self.assertTrue(cc.passed)
        self.assertGreaterEqual(cc.temporal_score, 0.80)
        self.assertEqual(cc.topology_score, 1.0)

        # 6. Remediation proposal and simulated execution
        self.assertIsNotNone(state.remediation_action)
        self.assertEqual(state.remediation_action.target_service, "checkoutservice")
        self.assertEqual(state.remediation_action.action_type, "scale_service")

        self.assertIsNotNone(state.remediation_result)
        self.assertTrue(state.remediation_result.success)

        # 7. Post-remediation recovery verification
        self.assertIsNotNone(state.verification_result)
        self.assertEqual(state.verification_result.final_status, "RESOLVED")
        self.assertFalse(state.verification_result.regression_detected)

    def test_investigation_default_blocks_low_confidence_remediation(self) -> None:
        # Default threshold is 0.80; diagnosis confidence is ~0.54
        engine = InvestigationEngine(budget=20)
        state = engine.investigate(self.case, window_mode="oracle")

        # Diagnosis succeeded but remediation is blocked by safety gate
        self.assertEqual(state.status, "DIAGNOSIS_COMPLETE")
        self.assertFalse(state.remediation_authorized)
        self.assertIsNotNone(state.remediation_result)
        self.assertFalse(state.remediation_result.success)
        self.assertTrue(any("SAFETY GATE BLOCKED" in l for l in state.remediation_result.execution_log))

        # Honest simulation: since remediation was blocked, verification confirms NOT_RESOLVED
        self.assertIsNotNone(state.verification_result)
        self.assertEqual(state.verification_result.final_status, "NOT_RESOLVED")

    def test_investigation_without_ground_truth(self) -> None:
        """Verify engine operates independently and successfully when ground truth root cause is hidden."""
        import copy

        # 1. Root cause label hidden, but injection time known (oracle mode)
        case_blind_label = copy.deepcopy(self.case)
        # Strip root cause and fault labels completely
        case_blind_label.ground_truth = type("GroundTruthProxy", (), {
            "values": {"inject_time": self.case.ground_truth.values["inject_time"]},
        })()

        engine = InvestigationEngine(budget=20)
        state = engine.investigate(case_blind_label, window_mode="oracle")
        self.assertIn(state.status, ("DIAGNOSIS_COMPLETE", "REMEDIATION_AUTHORIZED"))
        self.assertIsNotNone(state.decision)
        self.assertEqual(state.decision.root_cause_service, "checkoutservice")

        # 2. Entire ground truth stripped (ground_truth = None, detected window mode)
        case_no_gt = copy.deepcopy(self.case)
        case_no_gt.ground_truth = None
        state_no_gt = engine.investigate(case_no_gt)
        self.assertIn(state_no_gt.status, ("DIAGNOSIS_COMPLETE", "REMEDIATION_AUTHORIZED"))
        self.assertIsNotNone(state_no_gt.decision)
        self.assertGreater(len(state_no_gt.decision.ranked_candidates), 0)
        self.assertIsNotNone(state_no_gt.decision.root_cause_service)

    def test_query_selection_reflects_missing_modality_and_has_rationale(self) -> None:
        engine = InvestigationEngine(budget=20)
        state = engine.investigate(self.case)

        # Check queries executed
        self.assertGreater(len(state.queries_executed), 0)
        for q in state.queries_executed:
            self.assertTrue(len(q.rationale) > 0, f"Query {q.query_id} missing rationale")

        # Ensure no duplicate tool query on the same service
        seen_pairs = set()
        for q in state.queries_executed:
            pair = (q.service, q.tool_name)
            self.assertNotIn(pair, seen_pairs, f"Duplicate query detected: {pair}")
            seen_pairs.add(pair)

    def test_strict_budget_exhaustion(self) -> None:
        # Small budget of 3 points
        engine = InvestigationEngine(budget=3)
        state = engine.investigate(self.case, window_mode="oracle")

        self.assertLessEqual(state.total_query_cost, 3)
        self.assertGreaterEqual(state.remaining_budget, 0)
        self.assertEqual(state.total_query_cost + state.remaining_budget, 3)
        self.assertIn(state.status, ("DIAGNOSIS_COMPLETE", "REMEDIATION_AUTHORIZED", "BUDGET_EXHAUSTED"))


if __name__ == "__main__":
    unittest.main()
