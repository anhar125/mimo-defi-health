"""Monitor — Main orchestration loop for DeFi Health Monitoring.

Coordinates the Collector → Analyzer → Predictor → Alerter pipeline.
Runs on a configurable polling interval and manages agent lifecycles.

Usage:
    python -m mimo_defi_health.monitor
    python -m mimo_defi_health.monitor --interval 30 --protocols aave,lido
    python -m mimo_defi_health.monitor --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from mimo_defi_health.agents.alerter import AlerterAgent
from mimo_defi_health.agents.analyzer import AnalyzerAgent
from mimo_defi_health.agents.collector import CollectorAgent
from mimo_defi_health.agents.predictor import PredictorAgent
from mimo_defi_health.models import (
    AlertChannel,
    MonitoringConfig,
    ProtocolChain,
    RiskLevel,
)

logger = structlog.get_logger(__name__)


def _load_config_from_env() -> MonitoringConfig:
    """Load configuration from environment variables."""
    import os

    chains_str = os.getenv("MONITORED_CHAINS", "ethereum")
    chains = []
    for c in chains_str.split(","):
        try:
            chains.append(ProtocolChain(c.strip().lower()))
        except ValueError:
            logger.warning("unknown_chain", chain=c.strip())

    channels_str = os.getenv("ALERT_CHANNELS", "slack")
    channels = []
    for ch in channels_str.split(","):
        try:
            channels.append(AlertChannel(ch.strip().lower()))
        except ValueError:
            logger.warning("unknown_channel", channel=ch.strip())

    return MonitoringConfig(
        poll_interval_seconds=int(os.getenv("POLL_INTERVAL", "60")),
        protocols=os.getenv("PROTOCOLS", "aave,compound,makerdao,lido").split(","),
        chains=chains or [ProtocolChain.ETHEREUM],
        tvl_drop_threshold_pct=float(os.getenv("TVL_DROP_THRESHOLD_PCT", "5.0")),
        utilization_spike_threshold=float(os.getenv("UTILIZATION_SPIKE_THRESHOLD", "85.0")),
        yield_anomaly_zscore=float(os.getenv("YIELD_ANOMALY_ZSCORE", "2.5")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
        mimo_api_url=os.getenv("MIMO_API_URL", "https://api.mimo.xiaomi.com/v2.5"),
        mimo_max_reasoning_steps=int(os.getenv("MIMO_MAX_REASONING_STEPS", "15")),
        mimo_temperature=float(os.getenv("MIMO_TEMPERATURE", "0.1")),
        mimo_max_tokens=int(os.getenv("MIMO_MAX_TOKENS", "1024")),
        alert_channels=channels or [AlertChannel.SLACK],
        dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
    )


class DeFiHealthMonitor:
    """Main orchestrator for the DeFi health monitoring pipeline.

    Lifecycle:
    1. Initialize agents (Collector, Analyzer, Predictor, Alerter)
    2. On each poll cycle:
       a. Collect metrics from all configured protocols
       b. Analyze metrics for anomalies
       c. If anomalies found, send to MiMo for risk assessment
       d. Deliver alerts for medium+ risk assessments
    3. Graceful shutdown on SIGINT/SIGTERM
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config
        self._running = False
        self._cycle_count = 0
        self._total_tokens_used = 0

        # Initialize agents with config
        self._collector = CollectorAgent()
        self._analyzer = AnalyzerAgent(
            tvl_drop_threshold_pct=config.tvl_drop_threshold_pct,
            utilization_spike_threshold=config.utilization_spike_threshold,
            yield_anomaly_zscore=config.yield_anomaly_zscore,
        )

        import os

        self._predictor = PredictorAgent(
            api_key=os.getenv("MIMO_API_KEY", ""),
            api_url=config.mimo_api_url,
            max_reasoning_steps=config.mimo_max_reasoning_steps,
            temperature=config.mimo_temperature,
            max_tokens=config.mimo_max_tokens,
        )

        self._alerter = AlerterAgent(
            channels=config.alert_channels,
            slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            pagerduty_routing_key=os.getenv("PAGERDUTY_ROUTING_KEY", ""),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL", ""),
            dry_run=config.dry_run,
        )

    async def run(self) -> None:
        """Start the monitoring loop."""
        self._running = True
        self._setup_signal_handlers()

        logger.info(
            "monitor_start",
            protocols=self._config.protocols,
            poll_interval=self._config.poll_interval_seconds,
            channels=[c.value for c in self._config.alert_channels],
            dry_run=self._config.dry_run,
        )

        async with self._collector, self._predictor, self._alerter:
            while self._running:
                try:
                    await self._run_cycle()
                except Exception as e:
                    logger.error("cycle_error", error=str(e), cycle=self._cycle_count)

                if self._running:
                    await asyncio.sleep(self._config.poll_interval_seconds)

        logger.info(
            "monitor_stopped",
            cycles_completed=self._cycle_count,
            total_tokens_used=self._total_tokens_used,
        )

    async def _run_cycle(self) -> None:
        """Execute a single monitoring cycle."""
        self._cycle_count += 1
        cycle_start = datetime.now(timezone.utc)

        logger.info("cycle_start", cycle=self._cycle_count)

        # Step 1: Collect metrics
        logger.debug("step_collecting")
        metrics_list = await self._collector.collect_all(
            protocol_ids=self._config.protocols,
            chains=self._config.chains,
        )

        if not metrics_list:
            logger.warning("no_metrics_collected")
            return

        logger.info("metrics_collected", count=len(metrics_list))

        # Step 2: Analyze for anomalies
        logger.debug("step_analyzing")
        anomalies = self._analyzer.analyze_batch(metrics_list)

        # Step 3: MiMo risk assessment for protocols with anomalies
        assessed_protocols = set()
        if anomalies:
            logger.debug("step_predicting", anomaly_count=len(anomalies))

            # Group anomalies by protocol
            anomalies_by_protocol: dict[str, list] = {}
            for anomaly in anomalies:
                anomalies_by_protocol.setdefault(anomaly.protocol_id, []).append(anomaly)

            # Build metrics lookup
            metrics_by_protocol = {m.protocol_id: m for m in metrics_list}

            for protocol_id, protocol_anomalies in anomalies_by_protocol.items():
                if protocol_id not in metrics_by_protocol:
                    continue

                metrics = metrics_by_protocol[protocol_id]

                try:
                    # Fetch historical data for context
                    historical = await self._collector.get_historical_tvl(protocol_id, days=7)

                    assessment = await self._predictor.assess_risk(
                        metrics=metrics,
                        anomalies=protocol_anomalies,
                        historical_data=historical,
                    )

                    self._total_tokens_used += assessment.input_tokens + assessment.output_tokens
                    assessed_protocols.add(protocol_id)

                    # Step 4: Send alerts for medium+ risk
                    if assessment.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
                        await self._alerter.send_risk_alert(
                            assessment=assessment,
                            protocol_name=metrics.protocol_name,
                        )

                except Exception as e:
                    logger.error(
                        "prediction_failed",
                        protocol=protocol_id,
                        error=str(e),
                    )

        cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        logger.info(
            "cycle_complete",
            cycle=self._cycle_count,
            protocols_scanned=len(metrics_list),
            anomalies_found=len(anomalies),
            protocols_assessed=len(assessed_protocols),
            tokens_used=self._total_tokens_used,
            duration_seconds=round(cycle_duration, 2),
        )

    def _setup_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

    def _handle_shutdown(self) -> None:
        """Handle shutdown signal."""
        logger.info("shutdown_received")
        self._running = False

    def stop(self) -> None:
        """Programmatic stop."""
        self._running = False


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="DeFi Protocol Health Monitor — Powered by MiMo V2.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mimo_defi_health.monitor
  python -m mimo_defi_health.monitor --interval 30 --protocols aave,lido
  python -m mimo_defi_health.monitor --dry-run --log-level DEBUG
        """,
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--protocols", type=str, default=None,
        help="Comma-separated protocol slugs (default: aave,compound,makerdao,lido)",
    )
    parser.add_argument(
        "--chains", type=str, default=None,
        help="Comma-separated chains (default: ethereum)",
    )
    parser.add_argument(
        "--channels", type=str, default=None,
        help="Comma-separated alert channels (default: slack)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without sending alerts",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog.stdlib, "LogLevel", lambda x: True)(level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    # Configure logging
    configure_logging(args.log_level)

    # Load config from env, override with CLI args
    import os

    config = _load_config_from_env()

    if args.interval is not None:
        config.poll_interval_seconds = args.interval
    if args.protocols:
        config.protocols = args.protocols.split(",")
    if args.dry_run:
        config.dry_run = True

    # Validate API key
    if not os.getenv("MIMO_API_KEY"):
        logger.warning("MIMO_API_KEY not set — predictor will fail on API calls")

    # Run monitor
    monitor = DeFiHealthMonitor(config)

    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        logger.info("interrupted")


if __name__ == "__main__":
    main()
