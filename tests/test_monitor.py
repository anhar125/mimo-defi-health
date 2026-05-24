"""Tests for the Monitor orchestrator."""

from __future__ import annotations

import pytest

from mimo_defi_health.monitor import DeFiHealthMonitor, parse_args
from mimo_defi_health.models import MonitoringConfig, AlertChannel


class TestMonitorConfig:
    """Tests for monitor configuration."""

    def test_default_config(self) -> None:
        config = MonitoringConfig()
        monitor = DeFiHealthMonitor(config)
        assert monitor._config.poll_interval_seconds == 60
        assert monitor._cycle_count == 0
        assert monitor._total_tokens_used == 0

    def test_custom_config(self) -> None:
        config = MonitoringConfig(
            poll_interval_seconds=30,
            dry_run=True,
            protocols=["aave", "lido"],
        )
        monitor = DeFiHealthMonitor(config)
        assert monitor._config.poll_interval_seconds == 30
        assert monitor._config.dry_run is True
        assert len(monitor._config.protocols) == 2

    def test_monitor_stop(self) -> None:
        config = MonitoringConfig()
        monitor = DeFiHealthMonitor(config)
        assert monitor._running is False
        monitor._running = True
        monitor.stop()
        assert monitor._running is False


class TestCLIArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self) -> None:
        args = parse_args.__wrapped__() if hasattr(parse_args, '__wrapped__') else None
        # Just test that parse_args is callable and returns an argparse.Namespace
        import sys
        original_argv = sys.argv
        try:
            sys.argv = ["monitor"]
            args = parse_args()
            assert args.interval is None
            assert args.protocols is None
            assert args.dry_run is False
            assert args.log_level == "INFO"
        finally:
            sys.argv = original_argv

    def test_custom_args(self) -> None:
        import sys
        original_argv = sys.argv
        try:
            sys.argv = [
                "monitor",
                "--interval", "30",
                "--protocols", "aave,lido",
                "--dry-run",
                "--log-level", "DEBUG",
            ]
            args = parse_args()
            assert args.interval == 30
            assert args.protocols == "aave,lido"
            assert args.dry_run is True
            assert args.log_level == "DEBUG"
        finally:
            sys.argv = original_argv


class TestAgentInitialization:
    """Tests that agents are properly initialized."""

    def test_agents_created(self) -> None:
        config = MonitoringConfig(
            alert_channels=[AlertChannel.SLACK, AlertChannel.TELEGRAM],
        )
        monitor = DeFiHealthMonitor(config)
        assert monitor._collector is not None
        assert monitor._analyzer is not None
        assert monitor._predictor is not None
        assert monitor._alerter is not None

    def test_analyzer_thresholds_from_config(self) -> None:
        config = MonitoringConfig(
            tvl_drop_threshold_pct=3.0,
            utilization_spike_threshold=90.0,
            yield_anomaly_zscore=2.0,
        )
        monitor = DeFiHealthMonitor(config)
        assert monitor._analyzer._tvl_drop_threshold == 3.0
        assert monitor._analyzer._utilization_threshold == 90.0
        assert monitor._analyzer._yield_zscore_threshold == 2.0
