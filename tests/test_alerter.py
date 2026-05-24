"""Tests for AlerterAgent."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mimo_defi_health.agents.alerter import AlerterAgent
from mimo_defi_health.models import (
    AlertChannel,
    Anomaly,
    AnomalyType,
    PredictedEvent,
    RiskAssessment,
    RiskLevel,
    Severity,
)


def _make_assessment(
    risk_level: RiskLevel = RiskLevel.HIGH,
    protocol_id: str = "aave",
) -> RiskAssessment:
    return RiskAssessment(
        id="risk-test-001",
        protocol_id=protocol_id,
        risk_level=risk_level,
        confidence=0.85,
        summary="Elevated risk from TVL decline",
        root_cause="Whale withdrawals",
        reasoning_chain=["Step 1: TVL drop", "Step 2: Utilization spike"],
        predicted_events=[
            PredictedEvent(
                event_type="liquidation_cascade",
                probability=0.72,
                estimated_time_hours=3.0,
                impact_description="$50M in liquidations",
                mitigation="Monitor closely",
            ),
        ],
        recommendations=["Increase monitoring frequency"],
        input_tokens=2400,
        output_tokens=800,
        reasoning_steps=2,
    )


def _make_anomaly(protocol_id: str = "aave") -> Anomaly:
    return Anomaly(
        id="anomaly-001",
        protocol_id=protocol_id,
        anomaly_type=AnomalyType.TVL_DROP,
        severity=Severity.WARNING,
        description="TVL dropped 7.5%",
        metric_name="tvl_change_1h",
        metric_value=-7.5,
        baseline_value=0.0,
    )


class TestAlerterInit:
    """Tests for AlerterAgent initialization."""

    def test_default_init(self) -> None:
        alerter = AlerterAgent()
        assert AlertChannel.SLACK in alerter._channels
        assert alerter._dry_run is False

    def test_dry_run(self) -> None:
        alerter = AlerterAgent(dry_run=True)
        assert alerter._dry_run is True

    def test_multiple_channels(self) -> None:
        alerter = AlerterAgent(
            channels=[AlertChannel.SLACK, AlertChannel.TELEGRAM],
        )
        assert len(alerter._channels) == 2


class TestDryRun:
    """Tests for dry-run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_sends_nothing(self) -> None:
        alerter = AlerterAgent(
            channels=[AlertChannel.SLACK],
            dry_run=True,
        )
        async with alerter:
            assessment = _make_assessment()
            alerts = await alerter.send_risk_alert(assessment, "Aave")

            assert len(alerts) == 1
            assert alerts[0].delivered is True  # Logged as delivered in dry run
            assert alerts[0].delivered_at is not None


class TestLowRiskSkipped:
    """Tests that low risk assessments don't generate alerts."""

    @pytest.mark.asyncio
    async def test_low_risk_not_alerted(self) -> None:
        alerter = AlerterAgent(
            channels=[AlertChannel.SLACK],
            dry_run=True,
        )
        async with alerter:
            assessment = _make_assessment(risk_level=RiskLevel.LOW)
            alerts = await alerter.send_risk_alert(assessment)
            assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_medium_risk_alerted(self) -> None:
        alerter = AlerterAgent(
            channels=[AlertChannel.SLACK],
            dry_run=True,
        )
        async with alerter:
            assessment = _make_assessment(risk_level=RiskLevel.MEDIUM)
            alerts = await alerter.send_risk_alert(assessment)
            assert len(alerts) == 1

    @pytest.mark.asyncio
    async def test_critical_risk_alerted(self) -> None:
        alerter = AlerterAgent(
            channels=[AlertChannel.SLACK],
            dry_run=True,
        )
        async with alerter:
            assessment = _make_assessment(risk_level=RiskLevel.CRITICAL)
            alerts = await alerter.send_risk_alert(assessment)
            assert len(alerts) == 1


class TestFormatting:
    """Tests for alert message formatting."""

    def test_risk_message_format(self) -> None:
        assessment = _make_assessment()
        message = AlerterAgent._format_risk_message(assessment, "Aave")

        assert "HIGH" in message
        assert "Aave" in message
        assert "85%" in message  # confidence
        assert "Whale withdrawals" in message
        assert "liquidation_cascade" in message
        assert "Recommendations:" in message

    def test_risk_markdown_format(self) -> None:
        assessment = _make_assessment()
        markdown = AlerterAgent._format_risk_markdown(assessment, "Aave")

        assert "*Risk Level:*" in markdown
        assert "85%" in markdown
        assert "*Predicted Events:*" in markdown
        assert "*Recommendations:*" in markdown

    def test_risk_message_without_protocol_name(self) -> None:
        assessment = _make_assessment()
        message = AlerterAgent._format_risk_message(assessment, "")
        assert "aave" in message

    def test_anomaly_markdown_format(self) -> None:
        anomaly = _make_anomaly()
        markdown = AlerterAgent._format_anomaly_markdown(anomaly)

        assert "tvl_drop" in markdown
        assert "warning" in markdown
        assert "7.5" in markdown


class TestRiskToSeverity:
    """Tests for risk level to severity mapping."""

    def test_mapping(self) -> None:
        assert AlerterAgent._risk_to_severity(RiskLevel.LOW) == Severity.INFO
        assert AlerterAgent._risk_to_severity(RiskLevel.MEDIUM) == Severity.WARNING
        assert AlerterAgent._risk_to_severity(RiskLevel.HIGH) == Severity.CRITICAL
        assert AlerterAgent._risk_to_severity(RiskLevel.CRITICAL) == Severity.EMERGENCY
