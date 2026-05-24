"""Predictor Agent — MiMo V2.5 long-chain reasoning for DeFi risk prediction.

Sends anomaly data and historical context to Xiaomi MiMo V2.5 for
multi-step causal reasoning. The model traces threat chains and produces
structured risk assessments with predicted events and recommendations.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from mimo_defi_health.models import (
    Anomaly,
    PredictedEvent,
    ProtocolMetrics,
    RiskAssessment,
    RiskLevel,
)

logger = structlog.get_logger(__name__)

# ── MiMo System Prompt ──────────────────────────────────────────────────────

MIMO_SYSTEM_PROMPT = """You are an expert DeFi risk analyst with deep knowledge of:
- Lending protocols (Aave, Compound, MakerDAO)
- Liquid staking derivatives (Lido, Rocket Pool)
- AMM mechanics (Uniswap, Curve)
- Oracle systems (Chainlink, Pyth)
- MEV and sandwich attack patterns
- DeFi exploit history and attack vectors
- Cross-protocol contagion risks

Your task: Analyze the provided protocol metrics and anomalies to produce a
structured risk assessment. Follow this reasoning chain:

1. ANOMALY INTERPRETATION: What do the raw numbers mean in context?
2. ROOT CAUSE ANALYSIS: What underlying factors are causing these anomalies?
3. HISTORICAL PARALLEL: What past DeFi incidents does this resemble?
4. CONTAGION ASSESSMENT: Could this affect other protocols or chains?
5. TRAJECTORY PREDICTION: If unchecked, what happens next? (2-6 hour window)
6. CONFIDENCE CALIBRATION: How certain are you, and why?
7. RECOMMENDATIONS: What actions should be taken?

Be precise with numbers. Cite specific thresholds and precedents.
Output a JSON assessment following the schema provided."""

MIMO_USER_PROMPT = """Analyze the following DeFi protocol data and anomalies:

## Protocol: {protocol_name} ({protocol_id})
Chain: {chain}
Timestamp: {timestamp}

### Current Metrics
- TVL: ${tvl_total:,.0f}
- TVL Change (1h): {tvl_change_1h:+.2f}%
- TVL Change (24h): {tvl_change_24h:+.2f}%
- Utilization: {utilization_rate:.1f}%
- Supply APY: {supply_apy:.2f}%
- Borrow APY: {borrow_apy:.2f}%
- Health Factor (avg): {health_factor_avg:.3f}
- Liquidations (1h): {liquidation_count_1h}
- Active Positions: {active_positions}

### Detected Anomalies
{anomalies_text}

### Historical Context (last 24h)
{history_text}

Provide your analysis as JSON with this structure:
{{
    "risk_level": "low|medium|high|critical",
    "confidence": 0.0-1.0,
    "reasoning_chain": ["step 1", "step 2", ...],
    "summary": "brief risk summary",
    "root_cause": "identified root cause",
    "predicted_events": [
        {{
            "event_type": "string",
            "probability": 0.0-1.0,
            "estimated_time_hours": float,
            "impact_description": "string",
            "mitigation": "string"
        }}
    ],
    "recommendations": ["action 1", "action 2", ...]
}}"""


class PredictorAgent:
    """Uses MiMo V2.5 for long-chain reasoning on DeFi risk assessment.

    Sends structured prompts with protocol metrics and detected anomalies
    to the MiMo API, receives multi-step reasoning chains, and parses
    structured risk assessments.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.mimo.xiaomi.com/v2.5",
        max_reasoning_steps: int = 15,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url.rstrip("/")
        self._max_reasoning_steps = max_reasoning_steps
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PredictorAgent":
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── Public API ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        reraise=True,
    )
    async def assess_risk(
        self,
        metrics: ProtocolMetrics,
        anomalies: list[Anomaly],
        historical_data: list[dict[str, Any]] | None = None,
    ) -> RiskAssessment:
        """Perform MiMo V2.5 risk assessment on protocol data.

        Args:
            metrics: Current protocol metrics.
            anomalies: Detected anomalies to analyze.
            historical_data: Optional historical TVL/metrics data.

        Returns:
            Structured RiskAssessment with reasoning chain.
        """
        logger.info(
            "mimo_assessment_start",
            protocol=metrics.protocol_id,
            anomaly_count=len(anomalies),
        )

        prompt = self._build_prompt(metrics, anomalies, historical_data or [])
        response = await self._call_mimo(prompt)
        assessment = self._parse_response(response, metrics.protocol_id)

        logger.info(
            "mimo_assessment_complete",
            protocol=metrics.protocol_id,
            risk_level=assessment.risk_level.value,
            confidence=assessment.confidence,
            reasoning_steps=assessment.reasoning_steps,
            input_tokens=assessment.input_tokens,
            output_tokens=assessment.output_tokens,
        )

        return assessment

    # ── Internal Methods ────────────────────────────────────────────────

    def _build_prompt(
        self,
        metrics: ProtocolMetrics,
        anomalies: list[Anomaly],
        historical_data: list[dict[str, Any]],
    ) -> str:
        """Build the user prompt for MiMo V2.5."""
        anomalies_text = "\n".join(
            f"- **{a.anomaly_type.value.upper()}** (severity: {a.severity.value}): "
            f"{a.description}\n"
            f"  Metric: {a.metric_name} = {a.metric_value} "
            f"(baseline: {a.baseline_value}, deviation: {a.deviation_pct:.1f}%, "
            f"z-score: {a.z_score:.2f})"
            for a in anomalies
        ) if anomalies else "No anomalies detected."

        history_text = ""
        if historical_data:
            recent = historical_data[-10:]
            history_text = "\n".join(
                f"- {entry.get('timestamp', 'N/A')}: TVL=${entry.get('tvl', 0):,.0f}"
                for entry in recent
            )
        else:
            history_text = "No historical data available."

        return MIMO_USER_PROMPT.format(
            protocol_name=metrics.protocol_name,
            protocol_id=metrics.protocol_id,
            chain=metrics.chain.value,
            timestamp=metrics.timestamp.isoformat(),
            tvl_total=metrics.tvl_total,
            tvl_change_1h=metrics.tvl_change_1h,
            tvl_change_24h=metrics.tvl_change_24h,
            utilization_rate=metrics.utilization_rate,
            supply_apy=metrics.supply_apy,
            borrow_apy=metrics.borrow_apy,
            health_factor_avg=metrics.health_factor_avg,
            liquidation_count_1h=metrics.liquidation_count_1h,
            active_positions=metrics.active_positions,
            anomalies_text=anomalies_text,
            history_text=history_text,
        )

    async def _call_mimo(self, user_prompt: str) -> dict[str, Any]:
        """Call the MiMo V2.5 API endpoint."""
        assert self._client is not None, "PredictorAgent must be used as async context manager"

        payload = {
            "model": "mimo-v2.5-100t",
            "messages": [
                {"role": "system", "content": MIMO_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "response_format": {"type": "json_object"},
        }

        response = await self._client.post(
            f"{self._api_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(
        self,
        response: dict[str, Any],
        protocol_id: str,
    ) -> RiskAssessment:
        """Parse MiMo API response into a RiskAssessment."""
        choice = response.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "{}")

        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("mimo_parse_failed", raw_content=content[:500])
            parsed = self._fallback_parse(content)

        reasoning_chain = parsed.get("reasoning_chain", [])
        risk_level_str = parsed.get("risk_level", "medium").lower()

        risk_level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL,
        }

        predicted_events = []
        for event in parsed.get("predicted_events", []):
            predicted_events.append(PredictedEvent(
                event_type=event.get("event_type", "unknown"),
                probability=float(event.get("probability", 0.5)),
                estimated_time_hours=float(event.get("estimated_time_hours", 2.0)),
                impact_description=event.get("impact_description", ""),
                mitigation=event.get("mitigation", ""),
            ))

        return RiskAssessment(
            id=f"risk-{uuid.uuid4().hex[:12]}",
            protocol_id=protocol_id,
            risk_level=risk_level_map.get(risk_level_str, RiskLevel.MEDIUM),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning_chain=reasoning_chain,
            summary=parsed.get("summary", "Unable to parse MiMo response"),
            root_cause=parsed.get("root_cause", ""),
            predicted_events=predicted_events,
            recommendations=parsed.get("recommendations", []),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_steps=len(reasoning_chain),
        )

    @staticmethod
    def _fallback_parse(content: str) -> dict[str, Any]:
        """Best-effort parsing when JSON output fails."""
        risk_level = "medium"
        for level in ["critical", "high", "medium", "low"]:
            if level in content.lower():
                risk_level = level
                break

        return {
            "risk_level": risk_level,
            "confidence": 0.3,
            "reasoning_chain": [f"Raw response analysis: {content[:200]}"],
            "summary": content[:300],
            "root_cause": "",
            "predicted_events": [],
            "recommendations": ["Manual review required — MiMo output could not be parsed"],
        }
