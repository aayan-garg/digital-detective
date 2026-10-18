"""Unit tests for advanced RCA feature extractors: causal residuals and Drain3 log miner."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from digital_detective.features.causal_residuals import (
    CausalResidualResult,
    EntityCausalResiduals,
    extract_causal_residuals,
    parse_rcaeval_metric_identifier,
)
from digital_detective.features.log_scorer import (
    DRAIN3_AVAILABLE,
    LogScorerResult,
    score_logs_evidence,
)
from digital_detective.telemetry import (
    CaseMetadata,
    GroundTruth,
    ModalityProvenance,
    TelemetryCase,
    TelemetryModality,
)


def _make_dummy_case(
    metric_cols: dict[str, np.ndarray],
    timestamps: np.ndarray,
    logs_data: dict[str, list] | None = None,
) -> TelemetryCase:
    table_dict = {"time": timestamps}
    table_dict.update(metric_cols)
    arrow_metrics = pa.Table.from_pydict(table_dict)

    prov_metrics = ModalityProvenance(
        source_dataset="test",
        source_case="test_case_1",
        original_format="parquet",
        original_field_names=tuple(arrow_metrics.column_names),
        timestamp_field="time",
    )
    mod_metrics = TelemetryModality(
        raw_data=arrow_metrics,
        provenance=prov_metrics,
    )

    mod_logs = None
    if logs_data is not None:
        arrow_logs = pa.Table.from_pydict(logs_data)
        prov_logs = ModalityProvenance(
            source_dataset="test",
            source_case="test_case_1",
            original_format="parquet",
            original_field_names=tuple(arrow_logs.column_names),
            timestamp_field="timestamp",
            identity_fields=("container_name",),
        )
        mod_logs = TelemetryModality(
            raw_data=arrow_logs,
            provenance=prov_logs,
        )

    return TelemetryCase(
        metadata=CaseMetadata("test_case_1", "RE2-OB", "RE2", "ob", "online-boutique"),
        ground_truth=GroundTruth({"root_cause_service": "checkoutservice", "fault": "cpu"}),
        metrics=mod_metrics,
        logs=mod_logs,
        traces=None,
    )


class TestCausalResiduals:
    def test_metric_identifier_parsing(self):
        p1 = parse_rcaeval_metric_identifier("checkoutservice_cpu")
        assert p1.entity == "checkoutservice"
        assert p1.signal == "cpu"

        p2 = parse_rcaeval_metric_identifier("paymentservice_latency-90")
        assert p2.entity == "paymentservice"
        assert p2.signal == "latency-90"

        p3 = parse_rcaeval_metric_identifier("frontend_mem")
        assert p3.entity == "frontend"
        assert p3.signal == "mem"

    def test_causal_residual_extraction(self):
        n = 100
        ts = np.arange(1000, 1000 + n)
        # Service A has normal baseline then spikes after onset (ts = 1050)
        signal_a = np.ones(n) * 10.0
        signal_a[50:] = 100.0  # massive anomaly after ts=1050

        signal_b = np.ones(n) * 5.0
        signal_b[50:] = 6.0    # slight variation

        case = _make_dummy_case(
            {"checkoutservice_cpu": signal_a, "paymentservice_cpu": signal_b},
            ts,
        )

        res = extract_causal_residuals(case, candidate_universe=["checkoutservice", "paymentservice"], onset_timestamp=1050)
        assert isinstance(res, CausalResidualResult)
        assert "checkoutservice" in res.entity_residuals
        assert "paymentservice" in res.entity_residuals

        # checkoutservice must have a significantly higher residual z-score than paymentservice
        z_a = res.entity_residuals["checkoutservice"].mean_residual_z
        z_b = res.entity_residuals["paymentservice"].mean_residual_z
        assert z_a > z_b
        assert z_a > 10.0

    def test_missing_metrics_graceful(self):
        case = _make_dummy_case({}, np.array([]))
        res = extract_causal_residuals(case, candidate_universe=["serviceA"])
        assert res.total_metrics == 0
        assert res.entity_residuals["serviceA"].mean_residual_z == 0.0


class TestLogScorer:
    def test_log_scorer_novel_templates(self):
        # 10 baseline logs, then 5 error logs after onset ts=1050
        n_base = 20
        n_fault = 10
        ts_base = [1000 + i for i in range(n_base)]
        ts_fault = [1050 + i for i in range(n_fault)]

        ts_all = ts_base + ts_fault
        containers = ["checkoutservice"] * (n_base + n_fault)
        messages = ["Normal heartbeat request OK"] * n_base + [
            "ERROR: Connection refused to database port 5432"
        ] * n_fault

        case = _make_dummy_case(
            {"dummy_metric": np.zeros(len(ts_all))},
            np.array(ts_all),
            logs_data={"timestamp": ts_all, "container_name": containers, "message": messages},
        )

        log_res = score_logs_evidence(case, candidate_universe=["checkoutservice", "paymentservice"], onset_timestamp=1050)
        assert isinstance(log_res, LogScorerResult)
        assert "checkoutservice" in log_res.entity_evidence

        ck_ev = log_res.entity_evidence["checkoutservice"]
        assert ck_ev.novel_template_count >= 1
        assert ck_ev.error_level_count >= 1
        assert ck_ev.anomaly_score > 0.5

    def test_missing_logs_graceful(self):
        # Case with no logs modality
        case = _make_dummy_case({"metric_a": np.zeros(10)}, np.arange(10), logs_data=None)
        res = score_logs_evidence(case, candidate_universe=["svc1", "svc2"])
        assert res.total_logs_processed == 0
        assert res.entity_evidence["svc1"].anomaly_score == 0.0
        assert res.entity_evidence["svc2"].novel_template_count == 0
