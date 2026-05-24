"""Tests for data models and Pydantic contracts."""

from datetime import datetime, timezone

import pytest

from mimo_defi_health.models import (
    Alert,
    AlertChannel,
    Anomaly,
    AnomalyType,
    MonitoringConfig,
    PredictedEvent,
    ProtocolChain,
    ProtocolMetrics,
    RiskAssessment,
    RiskLevel,
    Severity,
)


class TestProtocolMetrics:
    """Tests for ProtocolMetrics model."""

    def test_create_basic_metrics(self) -> None:
        metrics = ProtocolMetrics(
            protocol_id="aave",
            protocol_name="Aave",
            chain=ProtocolChain.ETHEREUM,
            tvl_total=10_000_000_000.0,
        )
        assert metrics.protocol_id == "aave"
        assert metrics.tvl_total == 10_000_000_000.0
        assert metrics.chain == ProtocolChain.ETHEREUM
        assert isinstance(metrics.timestamp, datetime)

    def test_create_full_metrics(self) -> None:
        metrics = ProtocolMetrics(
            protocol_id="compound",
            protocol_name="Compound",
            chain=ProtocolChain.ETHEREUM,
            tvl_total=5_000_000_000.0,
            tvl_change_1h=-3.5,
            tvl_change_24h=-8.2,
            utilization_rate=78.5,
            supply_apy=4.2,
            borrow_apy=6.8,
            health_factor_avg=1.35,
            liquidation_count_1h=12,
            unique_users_24h=5000,
            active_positions=15000,
            avg_gas_gwei=25.0,
        )
        assert metrics.utilization_rate == 78.5
        assert metrics.liquidation_count_1h == 12
        assert metrics.tvl_change_1h == -3.5

    def test_default_values(self) -> None:
        metrics = ProtocolMetrics(
            protocol_id="lido",
            protocol_name="Lido",
            chain=ProtocolChain.ETHEREUM,
            tvl_total=15_000_000_000.0,
        )
        assert metrics.tvl_change_1h == 0.0
        assert metrics.utilization_rate == 0.0
        assert metrics.supply_apy == 0.0
        assert metrics.health_factor_avg == 1.0
        assert metrics.liquidation_count_1h == 0

    def test_negative_tvl_not_allowed(self) -> None:
        with pytest.raises(Exception):
            ProtocolMetrics(
                protocol_id="test",
                protocol_name="Test",
                chain=ProtocolChain.ETHEREUM,
                tvl_total=-1.0,
            )

    def test_utilization_bounds(self) -> None:
        with pytest.raises(Exception):
            ProtocolMetrics(
                protocol_id="test",
                protocol_name="Test",
                chain=ProtocolChain.ETHEREUM,
                tvl_total=1.0,
                utilization_rate=101.0,
            )

    def test_forbids_extra_fields(self) -> None:
        with pytest.raises(Exception):
            ProtocolMetrics(
                protocol_id="test",
                protocol_name="Test",
                chain=ProtocolChain.ETHEREUM,
                tvl_total=1.0,
                unknown_field="should_fail",
            )


class TestAnomaly:
    """Tests for Anomaly model."""

    def test_create_anomaly(self) -> None:
        anomaly = Anomaly(
            id="anomaly-001",
            protocol_id="aave",
            anomaly_type=AnomalyType.TVL_DROP,
            severity=Severity.WARNING,
            description="TVL dropped 7.5% in 1 hour",
            metric_name="tvl_change_1h",
            metric_value=-7.5,
            baseline_value=0.0,
            deviation_pct=7.5,
            z_score=-2.8,
        )
        assert anomaly.anomaly_type == AnomalyType.TVL_DROP
        assert anomaly.z_score == -2.8
        assert anomaly.acknowledged is False

    def test_anomaly_with_related_metrics(self) -> None:
        anomaly = Anomaly(
            id="anomaly-002",
            protocol_id="compound",
            anomaly_type=AnomalyType.UTILIZATION_HIGH,
            severity=Severity.CRITICAL,
            description="Utilization at 96%",
            metric_name="utilization_rate",
            metric_value=96.0,
            baseline_value=80.0,
            related_metrics={"borrow_apr": 12.5, "supply_apr": 4.2},
        )
        assert anomaly.related_metrics["borrow_apr"] == 12.5


class TestRiskAssessment:
    """Tests for RiskAssessment model."""

    def test_create_assessment(self) -> None:
        assessment = RiskAssessment(
            id="risk-001",
            protocol_id="aave",
            risk_level=RiskLevel.HIGH,
            confidence=0.85,
            summary="Elevated risk due to utilization spike",
            root_cause="Large withdrawals by whale addresses",
            reasoning_chain=[
                "Step 1: Observed 7.5% TVL drop",
                "Step 2: Utilization increased to 92%",
                "Step 3: Historical pattern matches pre-liquidation cascade",
            ],
            predicted_events=[
                PredictedEvent(
                    event_type="liquidation_cascade",
                    probability=0.72,
                    estimated_time_hours=3.0,
                    impact_description="Potential $50M in liquidations",
                    mitigation="Monitor health factors, prepare circuit breakers",
                ),
            ],
            recommendations=[
                "Increase monitoring frequency to 15s",
                "Alert whale depositors",
                "Prepare emergency liquidity injection",
            ],
            input_tokens=2400,
            output_tokens=800,
            reasoning_steps=3,
        )
        assert assessment.risk_level == RiskLevel.HIGH
        assert len(assessment.predicted_events) == 1
        assert assessment.predicted_events[0].probability == 0.72
        assert assessment.input_tokens == 2400

    def test_default_assessment_values(self) -> None:
        assessment = RiskAssessment(
            id="risk-002",
            protocol_id="test",
            risk_level=RiskLevel.LOW,
            confidence=0.6,
            summary="All clear",
        )
        assert assessment.reasoning_chain == []
        assert assessment.predicted_events == []
        assert assessment.recommendations == []
        assert assessment.reasoning_steps == 0


class TestMonitoringConfig:
    """Tests for MonitoringConfig model."""

    def test_default_config(self) -> None:
        config = MonitoringConfig()
        assert config.poll_interval_seconds == 60
        assert "aave" in config.protocols
        assert config.dry_run is False
        assert config.mimo_temperature == 0.1

    def test_custom_config(self) -> None:
        config = MonitoringConfig(
            poll_interval_seconds=30,
            protocols=["aave", "lido", "curve"],
            tvl_drop_threshold_pct=3.0,
            dry_run=True,
        )
        assert config.poll_interval_seconds == 30
        assert len(config.protocols) == 3
        assert config.tvl_drop_threshold_pct == 3.0

    def test_interval_minimum(self) -> None:
        with pytest.raises(Exception):
            MonitoringConfig(poll_interval_seconds=5)


class TestEnums:
    """Tests for enum types."""

    def test_severity_values(self) -> None:
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"
        assert Severity.EMERGENCY.value == "emergency"

    def test_anomaly_types(self) -> None:
        assert AnomalyType.TVL_DROP.value == "tvl_drop"
        assert AnomalyType.UTILIZATION_HIGH.value == "utilization_high"

    def test_risk_levels(self) -> None:
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"


class TestAlert:
    """Tests for Alert model."""

    def test_create_alert(self) -> None:
        alert = Alert(
            id="alert-001",
            protocol_id="aave",
            severity=Severity.WARNING,
            channel=AlertChannel.SLACK,
            title="⚠️ TVL Drop Detected",
            message="Aave TVL dropped 7.5% in 1 hour",
        )
        assert alert.delivered is False
        assert alert.delivered_at is None
        assert alert.delivery_error == ""
