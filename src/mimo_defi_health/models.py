"""Data models and Pydantic contracts for the DeFi Health Monitor."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class Severity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    TVL_DROP = "tvl_drop"
    TVL_SPIKE = "tvl_spike"
    UTILIZATION_HIGH = "utilization_high"
    YIELD_ANOMALY = "yield_anomaly"
    GAS_ANOMALY = "gas_anomaly"
    LIQUIDATION_CLUSTER = "liquidation_cluster"
    ORACLE_LAG = "oracle_lag"
    CORRELATION_BREAK = "correlation_break"


class AlertChannel(str, Enum):
    """Supported alert delivery channels."""
    SLACK = "slack"
    TELEGRAM = "telegram"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"


class RiskLevel(str, Enum):
    """Risk assessment levels from MiMo reasoning."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProtocolChain(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    POLYGON = "polygon"
    BASE = "base"
    BSC = "bsc"


# ── Core Data Models ───────────────────────────────────────────────────────

class ProtocolMetrics(BaseModel):
    """Real-time metrics snapshot for a single DeFi protocol."""
    protocol_id: str = Field(..., description="Unique protocol identifier (slug)")
    protocol_name: str = Field(..., description="Human-readable protocol name")
    chain: ProtocolChain = Field(..., description="Primary chain")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # TVL metrics
    tvl_total: float = Field(..., ge=0, description="Total Value Locked in USD")
    tvl_change_1h: float = Field(default=0.0, description="1-hour TVL change (%)")
    tvl_change_24h: float = Field(default=0.0, description="24-hour TVL change (%)")

    # Utilization metrics
    utilization_rate: float = Field(default=0.0, ge=0, le=100, description="Capital utilization %")
    borrow_utilization: float = Field(default=0.0, ge=0, le=100)
    supply_utilization: float = Field(default=0.0, ge=0, le=100)

    # Yield metrics
    supply_apy: float = Field(default=0.0, description="Supply APY (%)")
    borrow_apy: float = Field(default=0.0, description="Borrow APY (%)")
    reward_apy: float = Field(default=0.0, description="Incentive APY (%)")

    # Risk indicators
    health_factor_avg: float = Field(default=1.0, ge=0, description="Avg user health factor")
    liquidation_count_1h: int = Field(default=0, ge=0)
    unique_users_24h: int = Field(default=0, ge=0)
    active_positions: int = Field(default=0, ge=0)

    # Gas / network
    avg_gas_gwei: float = Field(default=0.0, ge=0)

    model_config = {"extra": "forbid"}


class Anomaly(BaseModel):
    """A detected anomaly in protocol metrics."""
    id: str = Field(..., description="Unique anomaly identifier")
    protocol_id: str
    anomaly_type: AnomalyType
    severity: Severity
    detected_at: datetime = Field(default_factory=datetime.utcnow)

    description: str = Field(..., description="Human-readable anomaly description")
    metric_name: str = Field(..., description="Which metric triggered the anomaly")
    metric_value: float = Field(..., description="The anomalous value")
    baseline_value: float = Field(..., description="Expected baseline value")
    deviation_pct: float = Field(..., description="Deviation from baseline (%)")
    z_score: float = Field(default=0.0, description="Z-score of the anomaly")

    # Context for MiMo reasoning
    related_metrics: dict[str, Any] = Field(default_factory=dict)
    historical_context: list[dict[str, Any]] = Field(default_factory=list)

    acknowledged: bool = Field(default=False)


class RiskAssessment(BaseModel):
    """MiMo V2.5 risk assessment output."""
    id: str = Field(..., description="Unique assessment identifier")
    protocol_id: str
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1, description="Model confidence score")
    assessed_at: datetime = Field(default_factory=datetime.utcnow)

    # Reasoning chain
    reasoning_chain: list[str] = Field(
        default_factory=list,
        description="Step-by-step MiMo reasoning hops",
    )
    summary: str = Field(..., description="Concise risk summary")
    root_cause: str = Field(default="", description="Identified root cause (if any)")

    # Predictions
    predicted_events: list[PredictedEvent] = Field(default_factory=list)
    time_horizon_hours: float = Field(default=2.0, description="Prediction window")

    # Recommendations
    recommendations: list[str] = Field(default_factory=list)

    # Token usage
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_steps: int = Field(default=0, ge=0)


class PredictedEvent(BaseModel):
    """A predicted future event from MiMo reasoning."""
    event_type: str = Field(..., description="Type of predicted event")
    probability: float = Field(ge=0, le=1, description="Event probability")
    estimated_time_hours: float = Field(..., description="Estimated hours until event")
    impact_description: str = Field(default="")
    mitigation: str = Field(default="")


class Alert(BaseModel):
    """An alert generated for delivery."""
    id: str = Field(..., description="Unique alert identifier")
    protocol_id: str
    severity: Severity
    channel: AlertChannel
    created_at: datetime = Field(default_factory=datetime.utcnow)

    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Full alert message")
    markdown_body: str = Field(default="", description="Formatted for channel")

    # Links
    anomaly_ids: list[str] = Field(default_factory=list)
    risk_assessment_id: str = Field(default="")

    delivered: bool = Field(default=False)
    delivered_at: datetime | None = None
    delivery_error: str = Field(default="")


class MonitoringConfig(BaseModel):
    """Configuration for the monitoring system."""
    poll_interval_seconds: int = Field(default=60, ge=10)
    protocols: list[str] = Field(default_factory=lambda: ["aave", "compound", "makerdao", "lido"])
    chains: list[ProtocolChain] = Field(default_factory=lambda: [ProtocolChain.ETHEREUM])

    # Anomaly detection
    tvl_drop_threshold_pct: float = Field(default=5.0, gt=0)
    utilization_spike_threshold: float = Field(default=85.0, gt=0, le=100)
    yield_anomaly_zscore: float = Field(default=2.5, gt=0)
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)

    # MiMo settings
    mimo_api_url: str = Field(default="https://api.mimo.xiaomi.com/v2.5")
    mimo_max_reasoning_steps: int = Field(default=15, ge=1)
    mimo_temperature: float = Field(default=0.1, ge=0, le=1)
    mimo_max_tokens: int = Field(default=1024, ge=256)

    # Alerting
    alert_channels: list[AlertChannel] = Field(default_factory=lambda: [AlertChannel.SLACK])
    dry_run: bool = Field(default=False)


# Rebuild forward refs for nested models
RiskAssessment.model_rebuild()
