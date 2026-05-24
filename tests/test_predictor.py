"""Tests for PredictorAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mimo_defi_health.agents.predictor import PredictorAgent
from mimo_defi_health.models import (
    Anomaly,
    AnomalyType,
    ProtocolChain,
    ProtocolMetrics,
    RiskLevel,
    Severity,
)


def _make_metrics(protocol_id: str = "aave") -> ProtocolMetrics:
    return ProtocolMetrics(
        protocol_id=protocol_id,
        protocol_name=protocol_id.title(),
        chain=ProtocolChain.ETHEREUM,
        tvl_total=10_000_000_000.0,
        tvl_change_1h=-7.5,
        utilization_rate=92.0,
    )


def _make_anomaly(protocol_id: str = "aave") -> Anomaly:
    return Anomaly(
        id="anomaly-test-001",
        protocol_id=protocol_id,
        anomaly_type=AnomalyType.TVL_DROP,
        severity=Severity.WARNING,
        description="TVL dropped 7.5% in 1 hour",
        metric_name="tvl_change_1h",
        metric_value=-7.5,
        baseline_value=0.0,
        deviation_pct=7.5,
        z_score=-2.8,
    )


class TestPredictorAgentInit:
    """Tests for PredictorAgent initialization."""

    def test_default_init(self) -> None:
        predictor = PredictorAgent(api_key="test-key")
        assert predictor._api_key == "test-key"
        assert predictor._temperature == 0.1
        assert predictor._max_tokens == 1024

    def test_custom_init(self) -> None:
        predictor = PredictorAgent(
            api_key="key",
            api_url="https://custom.api.com/v2.5",
            temperature=0.5,
            max_tokens=2048,
        )
        assert predictor._api_url == "https://custom.api.com/v2.5"
        assert predictor._temperature == 0.5
        assert predictor._max_tokens == 2048


class TestPromptBuilding:
    """Tests for prompt construction."""

    def test_build_prompt_with_anomalies(self) -> None:
        predictor = PredictorAgent(api_key="test")
        metrics = _make_metrics()
        anomalies = [_make_anomaly()]

        prompt = predictor._build_prompt(metrics, anomalies, [])
        assert "aave" in prompt
        assert "10,000,000,000" in prompt
        assert "TVL_DROP" in prompt
        assert "92.0%" in prompt

    def test_build_prompt_without_anomalies(self) -> None:
        predictor = PredictorAgent(api_key="test")
        metrics = _make_metrics()
        prompt = predictor._build_prompt(metrics, [], [])

        assert "No anomalies detected" in prompt

    def test_build_prompt_with_history(self) -> None:
        predictor = PredictorAgent(api_key="test")
        metrics = _make_metrics()
        history = [
            {"timestamp": "2024-01-01T00:00:00Z", "tvl": 9_500_000_000},
            {"timestamp": "2024-01-02T00:00:00Z", "tvl": 10_000_000_000},
        ]
        prompt = predictor._build_prompt(metrics, [], history)
        assert "9,500,000,000" in prompt


class TestResponseParsing:
    """Tests for MiMo response parsing."""

    def test_parse_valid_json_response(self) -> None:
        predictor = PredictorAgent(api_key="test")

        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "risk_level": "high",
                            "confidence": 0.85,
                            "reasoning_chain": [
                                "Observed 7.5% TVL drop",
                                "Utilization spiked to 92%",
                                "Pattern matches pre-liquidation cascade",
                            ],
                            "summary": "Elevated risk from rapid TVL decline",
                            "root_cause": "Whale withdrawals",
                            "predicted_events": [
                                {
                                    "event_type": "liquidation_cascade",
                                    "probability": 0.72,
                                    "estimated_time_hours": 3.0,
                                    "impact_description": "$50M in liquidations",
                                    "mitigation": "Increase monitoring frequency",
                                }
                            ],
                            "recommendations": [
                                "Alert team",
                                "Prepare circuit breakers",
                            ],
                        })
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 2400,
                "completion_tokens": 800,
            },
        }

        assessment = predictor._parse_response(response, "aave")

        assert assessment.protocol_id == "aave"
        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.confidence == 0.85
        assert len(assessment.reasoning_chain) == 3
        assert assessment.root_cause == "Whale withdrawals"
        assert len(assessment.predicted_events) == 1
        assert assessment.predicted_events[0].event_type == "liquidation_cascade"
        assert len(assessment.recommendations) == 2
        assert assessment.input_tokens == 2400
        assert assessment.output_tokens == 800
        assert assessment.reasoning_steps == 3

    def test_parse_invalid_json_response(self) -> None:
        predictor = PredictorAgent(api_key="test")

        response = {
            "choices": [{"message": {"content": "This is not JSON"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

        assessment = predictor._parse_response(response, "test")
        # Should use fallback parsing
        assert assessment.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert "Manual review" in assessment.recommendations[0]

    def test_parse_low_risk_response(self) -> None:
        predictor = PredictorAgent(api_key="test")

        response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "risk_level": "low",
                            "confidence": 0.95,
                            "reasoning_chain": ["All metrics normal"],
                            "summary": "Protocol operating normally",
                            "predicted_events": [],
                            "recommendations": [],
                        })
                    }
                }
            ],
            "usage": {},
        }

        assessment = predictor._parse_response(response, "lido")
        assert assessment.risk_level == RiskLevel.LOW
        assert assessment.confidence == 0.95


class TestFallbackParsing:
    """Tests for fallback parsing."""

    def test_fallback_detects_critical(self) -> None:
        result = PredictorAgent._fallback_parse(
            "Critical risk detected in the protocol"
        )
        assert result["risk_level"] == "critical"

    def test_fallback_detects_high(self) -> None:
        result = PredictorAgent._fallback_parse(
            "There is a high chance of liquidation"
        )
        assert result["risk_level"] == "high"

    def test_fallback_default_medium(self) -> None:
        result = PredictorAgent._fallback_parse("Some random text")
        assert result["risk_level"] == "medium"
