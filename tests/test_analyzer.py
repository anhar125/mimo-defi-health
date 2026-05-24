"""Tests for AnalyzerAgent."""

from __future__ import annotations

import pytest

from mimo_defi_health.agents.analyzer import AnalyzerAgent
from mimo_defi_health.models import (
    AnomalyType,
    ProtocolChain,
    ProtocolMetrics,
    Severity,
)


def _make_metrics(
    protocol_id: str = "aave",
    tvl_total: float = 10_000_000_000.0,
    tvl_change_1h: float = 0.0,
    utilization_rate: float = 75.0,
    supply_apy: float = 4.0,
    health_factor_avg: float = 1.5,
    liquidation_count_1h: int = 0,
) -> ProtocolMetrics:
    """Helper to create test metrics."""
    return ProtocolMetrics(
        protocol_id=protocol_id,
        protocol_name=protocol_id.title(),
        chain=ProtocolChain.ETHEREUM,
        tvl_total=tvl_total,
        tvl_change_1h=tvl_change_1h,
        utilization_rate=utilization_rate,
        supply_apy=supply_apy,
        health_factor_avg=health_factor_avg,
        liquidation_count_1h=liquidation_count_1h,
    )


class TestAnalyzerInitialization:
    """Tests for AnalyzerAgent initialization."""

    def test_default_thresholds(self) -> None:
        analyzer = AnalyzerAgent()
        assert analyzer._tvl_drop_threshold == 5.0
        assert analyzer._utilization_threshold == 85.0
        assert analyzer._yield_zscore_threshold == 2.5

    def test_custom_thresholds(self) -> None:
        analyzer = AnalyzerAgent(
            tvl_drop_threshold_pct=3.0,
            utilization_spike_threshold=90.0,
        )
        assert analyzer._tvl_drop_threshold == 3.0
        assert analyzer._utilization_threshold == 90.0


class TestThresholdRules:
    """Tests for threshold-based anomaly detection."""

    def test_tvl_drop_warning(self) -> None:
        analyzer = AnalyzerAgent(tvl_drop_threshold_pct=5.0)
        metrics = _make_metrics(tvl_change_1h=-7.5)
        anomalies = analyzer.analyze_batch([metrics])

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == AnomalyType.TVL_DROP
        assert anomalies[0].severity == Severity.WARNING
        assert anomalies[0].metric_value == -7.5

    def test_tvl_drop_critical(self) -> None:
        analyzer = AnalyzerAgent(tvl_drop_threshold_pct=5.0)
        metrics = _make_metrics(tvl_change_1h=-15.0)
        anomalies = analyzer.analyze_batch([metrics])

        assert len(anomalies) == 1
        assert anomalies[0].severity == Severity.CRITICAL

    def test_no_tvl_drop(self) -> None:
        analyzer = AnalyzerAgent(tvl_drop_threshold_pct=5.0)
        metrics = _make_metrics(tvl_change_1h=-2.0)
        anomalies = analyzer.analyze_batch([metrics])

        tvl_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.TVL_DROP]
        assert len(tvl_anomalies) == 0

    def test_utilization_high_warning(self) -> None:
        analyzer = AnalyzerAgent(utilization_spike_threshold=85.0)
        metrics = _make_metrics(utilization_rate=90.0)
        anomalies = analyzer.analyze_batch([metrics])

        util_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.UTILIZATION_HIGH]
        assert len(util_anomalies) == 1
        assert util_anomalies[0].severity == Severity.WARNING

    def test_utilization_critical(self) -> None:
        analyzer = AnalyzerAgent(utilization_spike_threshold=85.0)
        metrics = _make_metrics(utilization_rate=97.0)
        anomalies = analyzer.analyze_batch([metrics])

        util_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.UTILIZATION_HIGH]
        assert len(util_anomalies) == 1
        assert util_anomalies[0].severity == Severity.CRITICAL

    def test_no_anomalies_in_normal_conditions(self) -> None:
        analyzer = AnalyzerAgent()
        metrics = _make_metrics(
            tvl_change_1h=-1.0,
            utilization_rate=60.0,
        )
        anomalies = analyzer.analyze_batch([metrics])
        assert len(anomalies) == 0


class TestBaselineComputation:
    """Tests for rolling baseline computation."""

    def test_baseline_empty_history(self) -> None:
        analyzer = AnalyzerAgent()
        baseline = analyzer.get_baseline("unknown-protocol")
        assert baseline is None

    def test_baseline_updated_after_analysis(self) -> None:
        analyzer = AnalyzerAgent()

        # Feed several data points
        for i in range(5):
            metrics = _make_metrics(tvl_total=10_000_000_000.0 + i * 100_000_000)
            analyzer.analyze_batch([metrics])

        baseline = analyzer.get_baseline("aave")
        assert baseline is not None
        assert "tvl_total" in baseline
        assert baseline["tvl_total"] > 0


class TestZScoreDetection:
    """Tests for statistical Z-score based detection."""

    def test_tvl_spike_detected(self) -> None:
        analyzer = AnalyzerAgent()

        # Establish baseline with stable data
        for _ in range(10):
            metrics = _make_metrics(tvl_total=10_000_000_000.0)
            analyzer.analyze_batch([metrics])

        # Now inject a spike
        spike_metrics = _make_metrics(tvl_total=15_000_000_000.0)
        anomalies = analyzer.analyze_batch([spike_metrics])

        spike_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.TVL_SPIKE]
        assert len(spike_anomalies) >= 1

    def test_multiple_protocols(self) -> None:
        analyzer = AnalyzerAgent()

        metrics = [
            _make_metrics(protocol_id="aave", tvl_change_1h=-8.0),
            _make_metrics(protocol_id="compound", utilization_rate=92.0),
            _make_metrics(protocol_id="lido"),
        ]
        anomalies = analyzer.analyze_batch(metrics)

        protocols_with_anomalies = {a.protocol_id for a in anomalies}
        assert "aave" in protocols_with_anomalies
        assert "compound" in protocols_with_anomalies
        assert "lido" not in protocols_with_anomalies


class TestHealthFactor:
    """Tests for health factor anomaly detection."""

    def test_low_health_factor_warning(self) -> None:
        analyzer = AnalyzerAgent()
        # Need at least 3 data points for z-score checks
        for _ in range(3):
            metrics = _make_metrics(health_factor_avg=1.5)
            analyzer.analyze_batch([metrics])

        low_hf = _make_metrics(health_factor_avg=1.05)
        anomalies = analyzer.analyze_batch([low_hf])

        hf_anomalies = [a for a in anomalies if a.metric_name == "health_factor_avg"]
        assert len(hf_anomalies) >= 1
        assert hf_anomalies[0].severity == Severity.WARNING

    def test_critical_health_factor(self) -> None:
        analyzer = AnalyzerAgent()
        for _ in range(3):
            metrics = _make_metrics(health_factor_avg=1.5)
            analyzer.analyze_batch([metrics])

        critical_hf = _make_metrics(health_factor_avg=0.95)
        anomalies = analyzer.analyze_batch([critical_hf])

        hf_anomalies = [a for a in anomalies if a.metric_name == "health_factor_avg"]
        assert len(hf_anomalies) >= 1
        assert hf_anomalies[0].severity == Severity.CRITICAL
