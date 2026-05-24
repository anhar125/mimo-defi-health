"""Alerter Agent — Multi-channel alert delivery for DeFi risk events.

Formats and delivers alerts through Slack, Telegram, PagerDuty, and
custom webhooks based on risk assessments and anomaly severity.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog

from mimo_defi_health.models import (
    Alert,
    AlertChannel,
    RiskAssessment,
    RiskLevel,
    Severity,
)

logger = structlog.get_logger(__name__)

# ── Severity Emoji/Formatting ───────────────────────────────────────────────

SEVERITY_EMOJI = {
    Severity.INFO: "ℹ️",
    Severity.WARNING: "⚠️",
    Severity.CRITICAL: "🔴",
    Severity.EMERGENCY: "🚨",
}

RISK_EMOJI = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.HIGH: "🟠",
    RiskLevel.CRITICAL: "🔴",
}


class AlerterAgent:
    """Delivers formatted alerts through multiple channels.

    Supports:
    - Slack (via incoming webhooks)
    - Telegram (via Bot API)
    - PagerDuty (via Events API v2)
    - Generic webhooks (JSON POST)
    """

    def __init__(
        self,
        channels: list[AlertChannel] | None = None,
        slack_webhook_url: str = "",
        telegram_bot_token: str = "",
        telegram_chat_id: str = "",
        pagerduty_routing_key: str = "",
        webhook_url: str = "",
        dry_run: bool = False,
    ) -> None:
        self._channels = channels or [AlertChannel.SLACK]
        self._slack_webhook = slack_webhook_url
        self._telegram_token = telegram_bot_token
        self._telegram_chat_id = telegram_chat_id
        self._pagerduty_key = pagerduty_routing_key
        self._webhook_url = webhook_url
        self._dry_run = dry_run
        self._client: httpx.AsyncClient | None = None
        self._sent_alerts: list[Alert] = []

    async def __aenter__(self) -> "AlerterAgent":
        self._client = httpx.AsyncClient(timeout=15.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── Public API ──────────────────────────────────────────────────────

    async def send_risk_alert(
        self,
        assessment: RiskAssessment,
        protocol_name: str = "",
    ) -> list[Alert]:
        """Send alerts for a risk assessment through all configured channels.

        Only sends alerts for MEDIUM or higher risk levels.

        Args:
            assessment: MiMo V2.5 risk assessment.
            protocol_name: Human-readable protocol name.

        Returns:
            List of Alert objects with delivery status.
        """
        if assessment.risk_level == RiskLevel.LOW:
            logger.debug("skipping_low_risk", protocol=assessment.protocol_id)
            return []

        alerts: list[Alert] = []
        for channel in self._channels:
            alert = await self._send_to_channel(
                channel=channel,
                assessment=assessment,
                protocol_name=protocol_name,
            )
            alerts.append(alert)

        return alerts

    async def send_anomaly_alert(
        self,
        anomaly: Any,  # Anomaly type
        channel: AlertChannel = AlertChannel.SLACK,
    ) -> Alert:
        """Send a raw anomaly alert (for anomalies without MiMo assessment)."""
        alert = Alert(
            id=f"alert-{uuid.uuid4().hex[:12]}",
            protocol_id=anomaly.protocol_id,
            severity=anomaly.severity,
            channel=channel,
            title=f"{SEVERITY_EMOJI.get(anomaly.severity, '❓')} "
                  f"{anomaly.anomaly_type.value.replace('_', ' ').title()}",
            message=anomaly.description,
            markdown_body=self._format_anomaly_markdown(anomaly),
            anomaly_ids=[anomaly.id],
        )

        await self._deliver(alert)
        return alert

    # ── Channel Dispatch ────────────────────────────────────────────────

    async def _send_to_channel(
        self,
        channel: AlertChannel,
        assessment: RiskAssessment,
        protocol_name: str,
    ) -> Alert:
        """Route alert to the appropriate channel."""
        severity = self._risk_to_severity(assessment.risk_level)
        risk_emoji = RISK_EMOJI.get(assessment.risk_level, "❓")

        title = (
            f"{risk_emoji} DeFi Risk Alert: {protocol_name or assessment.protocol_id} "
            f"— {assessment.risk_level.value.upper()}"
        )

        body = self._format_risk_message(assessment, protocol_name)
        markdown = self._format_risk_markdown(assessment, protocol_name)

        alert = Alert(
            id=f"alert-{uuid.uuid4().hex[:12]}",
            protocol_id=assessment.protocol_id,
            severity=severity,
            channel=channel,
            title=title,
            message=body,
            markdown_body=markdown,
            risk_assessment_id=assessment.id,
        )

        await self._deliver(alert)
        return alert

    async def _deliver(self, alert: Alert) -> None:
        """Deliver an alert through its assigned channel."""
        if self._dry_run:
            logger.info(
                "alert_dry_run",
                alert_id=alert.id,
                channel=alert.channel.value,
                title=alert.title,
            )
            alert.delivered = True
            alert.delivered_at = datetime.now(timezone.utc)
            self._sent_alerts.append(alert)
            return

        try:
            if alert.channel == AlertChannel.SLACK:
                await self._send_slack(alert)
            elif alert.channel == AlertChannel.TELEGRAM:
                await self._send_telegram(alert)
            elif alert.channel == AlertChannel.PAGERDUTY:
                await self._send_pagerduty(alert)
            elif alert.channel == AlertChannel.WEBHOOK:
                await self._send_webhook(alert)

            alert.delivered = True
            alert.delivered_at = datetime.now(timezone.utc)
            self._sent_alerts.append(alert)

            logger.info(
                "alert_delivered",
                alert_id=alert.id,
                channel=alert.channel.value,
                severity=alert.severity.value,
            )

        except Exception as e:
            alert.delivery_error = str(e)
            logger.error(
                "alert_delivery_failed",
                alert_id=alert.id,
                channel=alert.channel.value,
                error=str(e),
            )

    # ── Channel Implementations ─────────────────────────────────────────

    async def _send_slack(self, alert: Alert) -> None:
        """Send alert via Slack incoming webhook."""
        assert self._client is not None

        payload = {
            "text": alert.title,
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": alert.title, "emoji": True},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": alert.markdown_body or alert.message},
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Alert ID: `{alert.id}` | {alert.created_at.isoformat()}"},
                    ],
                },
            ],
        }

        response = await self._client.post(self._slack_webhook, json=payload)
        response.raise_for_status()

    async def _send_telegram(self, alert: Alert) -> None:
        """Send alert via Telegram Bot API."""
        assert self._client is not None

        text = f"*{alert.title}*\n\n{alert.markdown_body or alert.message}"
        text += f"\n\n`Alert ID: {alert.id}`"

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        payload = {
            "chat_id": self._telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()

    async def _send_pagerduty(self, alert: Alert) -> None:
        """Send alert via PagerDuty Events API v2."""
        assert self._client is not None

        severity_map = {
            Severity.INFO: "info",
            Severity.WARNING: "warning",
            Severity.CRITICAL: "critical",
            Severity.EMERGENCY: "critical",
        }

        payload = {
            "routing_key": self._pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": alert.title,
                "severity": severity_map.get(alert.severity, "warning"),
                "source": "mimo-defi-health",
                "component": alert.protocol_id,
                "custom_details": {
                    "message": alert.message,
                    "alert_id": alert.id,
                },
            },
        }

        response = await self._client.post(
            "https://events.pagerduty.com/v2/enqueue",
            json=payload,
        )
        response.raise_for_status()

    async def _send_webhook(self, alert: Alert) -> None:
        """Send alert via generic webhook (JSON POST)."""
        assert self._client is not None

        payload = {
            "alert_id": alert.id,
            "protocol_id": alert.protocol_id,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "markdown": alert.markdown_body,
            "timestamp": alert.created_at.isoformat(),
        }

        response = await self._client.post(self._webhook_url, json=payload)
        response.raise_for_status()

    # ── Formatting ──────────────────────────────────────────────────────

    @staticmethod
    def _format_risk_message(assessment: RiskAssessment, protocol_name: str) -> str:
        """Format a plain-text risk alert message."""
        lines = [
            f"Risk Level: {assessment.risk_level.value.upper()} "
            f"(confidence: {assessment.confidence:.0%})",
            f"Protocol: {protocol_name or assessment.protocol_id}",
            "",
            f"Summary: {assessment.summary}",
        ]

        if assessment.root_cause:
            lines.append(f"Root Cause: {assessment.root_cause}")

        if assessment.predicted_events:
            lines.append("")
            lines.append("Predicted Events:")
            for event in assessment.predicted_events:
                lines.append(
                    f"  • {event.event_type} "
                    f"(probability: {event.probability:.0%}, "
                    f"eta: {event.estimated_time_hours:.1f}h)"
                )

        if assessment.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for i, rec in enumerate(assessment.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        if assessment.reasoning_chain:
            lines.append("")
            lines.append(f"Reasoning ({assessment.reasoning_steps} steps):")
            for i, step in enumerate(assessment.reasoning_chain[:5], 1):
                lines.append(f"  Step {i}: {step}")

        return "\n".join(lines)

    @staticmethod
    def _format_risk_markdown(assessment: RiskAssessment, protocol_name: str) -> str:
        """Format a Markdown risk alert for Slack/Telegram."""
        risk_emoji = RISK_EMOJI.get(assessment.risk_level, "❓")

        lines = [
            f"*{risk_emoji} Risk Level:* `{assessment.risk_level.value.upper()}` "
            f"(confidence: {assessment.confidence:.0%})",
            f"*Protocol:* {protocol_name or assessment.protocol_id}",
            "",
            f"*Summary:* {assessment.summary}",
        ]

        if assessment.root_cause:
            lines.append(f"*Root Cause:* {assessment.root_cause}")

        if assessment.predicted_events:
            lines.append("")
            lines.append("*⚡ Predicted Events:*")
            for event in assessment.predicted_events:
                lines.append(
                    f"  • `{event.event_type}` — {event.probability:.0%} chance "
                    f"in {event.estimated_time_hours:.1f}h"
                )
                if event.mitigation:
                    lines.append(f"    _Mitigation: {event.mitigation}_")

        if assessment.recommendations:
            lines.append("")
            lines.append("*📋 Recommendations:*")
            for i, rec in enumerate(assessment.recommendations, 1):
                lines.append(f"  {i}. {rec}")

        return "\n".join(lines)

    @staticmethod
    def _format_anomaly_markdown(anomaly: Any) -> str:
        """Format a raw anomaly as Markdown."""
        return (
            f"*Type:* `{anomaly.anomaly_type.value}`\n"
            f"*Severity:* `{anomaly.severity.value}`\n"
            f"*Description:* {anomaly.description}\n"
            f"*Metric:* {anomaly.metric_name} = {anomaly.metric_value} "
            f"(baseline: {anomaly.baseline_value})"
        )

    @staticmethod
    def _risk_to_severity(risk_level: RiskLevel) -> Severity:
        """Map risk level to alert severity."""
        return {
            RiskLevel.LOW: Severity.INFO,
            RiskLevel.MEDIUM: Severity.WARNING,
            RiskLevel.HIGH: Severity.CRITICAL,
            RiskLevel.CRITICAL: Severity.EMERGENCY,
        }.get(risk_level, Severity.WARNING)
