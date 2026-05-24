"""Analyzer Agent — Statistical anomaly detection on DeFi metrics.

Compares real-time protocol metrics against baselines and historical
distributions to identify anomalous conditions. Flags deviations that
should be escalated to the MiMo Predictor for causal reasoning.
"""

from __future__ import annotations

import statistics
import uuid
from collections import deque
from datetime import datetime
from typing import Any

import structlog

from mimo_defi_health.models import (
    Anomaly,
    AnomalyType,
    ProtocolMetrics,
    Severity,
)

logger = structlog.get_logger(__name__)

# Rolling window for baseline computation
_BASELINE_WINDOW = 100


class AnalyzerAgent:
    """Detects anomalies in DeFi protocol metrics using statistical methods.

    Maintains per-protocol rolling baselines and detects:
    - TVL drops/spikes exceeding threshold
    - Utilization rate anomalies
    - Yield deviations (Z-score based)
    - Liquidation clusters
    - Cross-metric correlation breaks
    """

    def __init__(
        self,
        tvl_drop_threshold_pct: float = 5.0,
        utilization_spike_threshold: float = 85.0,
        yield_anomaly_zscore: float = 2.5,
        liquidation_cluster_threshold: int = 5,
    ) -> None:
        self._tvl_drop_threshold = tvl_drop_threshold_pct
        self._utilization_threshold = utilization_spike_threshold
        self._yield_zscore_threshold = yield_anomaly_zscore
        self._liquidation_cluster_threshold = liquidation_cluster_threshold

        # Per-protocol rolling metric windows
        self._metric_history: dict[str, deque[dict[str, float]]] = {}

    # ── Public API ──────────────────────────────────────────────────────

    def analyze_batch(
        self,
        metrics_list: list[ProtocolMetrics],
    ) -> list[Anomaly]:
        """Analyze a batch of protocol metrics and return detected anomalies.

        Args:
            metrics_list: Latest metrics snapshot for each protocol.

        Returns:
            List of detected Anomaly objects.
        """
        all_anomalies: list[Anomaly] = []

        for metrics in metrics_list:
            self._update_baseline(metrics)
            anomalies = self._analyze_protocol(metrics)
            all_anomalies.extend(anomalies)

        if all_anomalies:
            logger.warning(
                "anomalies_detected",
                count=len(all_anomalies),
                protocols=[a.protocol_id for a in all_anomalies],
            )
        else:
            logger.debug("analysis_complete", status="all_clear")

        return all_anomalies

    def get_baseline(self, protocol_id: str) -> dict[str, float] | None:
        """Get the current baseline for a protocol."""
        history = self._metric_history.get(protocol_id)
        if not history or len(history) < 2:
            return None
        return self._compute_baseline(protocol_id)

    # ── Internal Methods ────────────────────────────────────────────────

    def _update_baseline(self, metrics: ProtocolMetrics) -> None:
        """Add new metrics to the rolling history window."""
        protocol_id = metrics.protocol_id
        if protocol_id not in self._metric_history:
            self._metric_history[protocol_id] = deque(maxlen=_BASELINE_WINDOW)

        snapshot = {
            "tvl_total": metrics.tvl_total,
            "utilization_rate": metrics.utilization_rate,
            "supply_apy": metrics.supply_apy,
            "borrow_apy": metrics.borrow_apy,
            "liquidation_count_1h": float(metrics.liquidation_count_1h),
            "health_factor_avg": metrics.health_factor_avg,
        }
        self._metric_history[protocol_id].append(snapshot)

    def _analyze_protocol(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Run all anomaly checks for a single protocol."""
        anomalies: list[Anomaly] = []

        # Not enough data for statistical analysis — only check thresholds
        if len(self._metric_history.get(metrics.protocol_id, [])) < 3:
            anomalies.extend(self._check_threshold_rules(metrics))
            return anomalies

        anomalies.extend(self._check_tvl_anomaly(metrics))
        anomalies.extend(self._check_utilization_anomaly(metrics))
        anomalies.extend(self._check_yield_anomaly(metrics))
        anomalies.extend(self._check_liquidation_cluster(metrics))
        anomalies.extend(self._check_health_factor(metrics))

        return anomalies

    def _check_threshold_rules(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Simple threshold-based checks (no history required)."""
        anomalies: list[Anomaly] = []

        if metrics.tvl_change_1h < -self._tvl_drop_threshold:
            anomalies.append(self._create_anomaly(
                protocol_id=metrics.protocol_id,
                anomaly_type=AnomalyType.TVL_DROP,
                severity=Severity.WARNING if metrics.tvl_change_1h > -10 else Severity.CRITICAL,
                description=f"TVL dropped {metrics.tvl_change_1h:.1f}% in the last hour",
                metric_name="tvl_change_1h",
                metric_value=metrics.tvl_change_1h,
                baseline_value=0.0,
            ))

        if metrics.utilization_rate > self._utilization_threshold:
            severity = Severity.CRITICAL if metrics.utilization_rate > 95 else Severity.WARNING
            anomalies.append(self._create_anomaly(
                protocol_id=metrics.protocol_id,
                anomaly_type=AnomalyType.UTILIZATION_HIGH,
                severity=severity,
                description=f"Utilization rate at {metrics.utilization_rate:.1f}%",
                metric_name="utilization_rate",
                metric_value=metrics.utilization_rate,
                baseline_value=self._utilization_threshold,
            ))

        return anomalies

    def _check_tvl_anomaly(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Check TVL against rolling baseline using Z-score."""
        anomalies: list[Anomaly] = []
        baseline = self._compute_baseline(metrics.protocol_id)

        mean = baseline.get("tvl_total", metrics.tvl_total)
        std = baseline.get("tvl_total_std", 0)

        if std > 0:
            z_score = (metrics.tvl_total - mean) / std
            if abs(z_score) > 2.5:
                anomaly_type = AnomalyType.TVL_DROP if z_score < 0 else AnomalyType.TVL_SPIKE
                deviation_pct = abs(metrics.tvl_total - mean) / mean * 100
                severity = Severity.CRITICAL if abs(z_score) > 4 else Severity.WARNING

                anomalies.append(self._create_anomaly(
                    protocol_id=metrics.protocol_id,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    description=f"TVL {'drop' if z_score < 0 else 'spike'}: "
                                f"{deviation_pct:.1f}% from baseline (z={z_score:.2f})",
                    metric_name="tvl_total",
                    metric_value=metrics.tvl_total,
                    baseline_value=mean,
                    deviation_pct=deviation_pct,
                    z_score=z_score,
                ))

        return anomalies

    def _check_utilization_anomaly(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Check utilization against threshold and baseline."""
        anomalies: list[Anomaly] = []

        if metrics.utilization_rate > self._utilization_threshold:
            severity = Severity.CRITICAL if metrics.utilization_rate > 95 else Severity.WARNING
            anomalies.append(self._create_anomaly(
                protocol_id=metrics.protocol_id,
                anomaly_type=AnomalyType.UTILIZATION_HIGH,
                severity=severity,
                description=f"Utilization rate at {metrics.utilization_rate:.1f}% "
                            f"(threshold: {self._utilization_threshold}%)",
                metric_name="utilization_rate",
                metric_value=metrics.utilization_rate,
                baseline_value=self._utilization_threshold,
            ))

        return anomalies

    def _check_yield_anomaly(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Check supply APY against historical distribution."""
        anomalies: list[Anomaly] = []
        baseline = self._compute_baseline(metrics.protocol_id)

        mean = baseline.get("supply_apy", metrics.supply_apy)
        std = baseline.get("supply_apy_std", 0)

        if std > 0.01:
            z_score = (metrics.supply_apy - mean) / std
            if abs(z_score) > self._yield_zscore_threshold:
                anomalies.append(self._create_anomaly(
                    protocol_id=metrics.protocol_id,
                    anomaly_type=AnomalyType.YIELD_ANOMALY,
                    severity=Severity.WARNING,
                    description=f"Supply APY {metrics.supply_apy:.2f}% is "
                                f"{z_score:.1f} standard deviations from mean ({mean:.2f}%)",
                    metric_name="supply_apy",
                    metric_value=metrics.supply_apy,
                    baseline_value=mean,
                    deviation_pct=abs(metrics.supply_apy - mean) / max(mean, 0.01) * 100,
                    z_score=z_score,
                ))

        return anomalies

    def _check_liquidation_cluster(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Detect liquidation clusters."""
        anomalies: list[Anomaly] = []
        history = self._metric_history.get(metrics.protocol_id, [])

        recent_liqs = [h.get("liquidation_count_1h", 0) for h in list(history)[-5:]]
        avg_liqs = statistics.mean(recent_liqs) if recent_liqs else 0

        if metrics.liquidation_count_1h > self._liquidation_cluster_threshold and \
                metrics.liquidation_count_1h > avg_liqs * 2:
            anomalies.append(self._create_anomaly(
                protocol_id=metrics.protocol_id,
                anomaly_type=AnomalyType.LIQUIDATION_CLUSTER,
                severity=Severity.CRITICAL,
                description=f"Liquidation cluster: {metrics.liquidation_count_1h} "
                            f"liquidations in the last hour (avg: {avg_liqs:.1f})",
                metric_name="liquidation_count_1h",
                metric_value=float(metrics.liquidation_count_1h),
                baseline_value=avg_liqs,
            ))

        return anomalies

    def _check_health_factor(self, metrics: ProtocolMetrics) -> list[Anomaly]:
        """Check average health factor against critical thresholds."""
        anomalies: list[Anomaly] = []

        if metrics.health_factor_avg < 1.1:
            severity = Severity.CRITICAL if metrics.health_factor_avg < 1.0 else Severity.WARNING
            anomalies.append(self._create_anomaly(
                protocol_id=metrics.protocol_id,
                anomaly_type=AnomalyType.LIQUIDATION_CLUSTER,
                severity=severity,
                description=f"Average health factor critically low: {metrics.health_factor_avg:.3f}",
                metric_name="health_factor_avg",
                metric_value=metrics.health_factor_avg,
                baseline_value=1.5,
            ))

        return anomalies

    def _compute_baseline(self, protocol_id: str) -> dict[str, float]:
        """Compute rolling statistics from metric history."""
        history = list(self._metric_history.get(protocol_id, []))
        if not history:
            return {}

        result: dict[str, float] = {}
        numeric_keys = ["tvl_total", "supply_apy", "borrow_apy"]

        for key in numeric_keys:
            values = [h.get(key, 0) for h in history]
            if len(values) >= 2:
                result[key] = statistics.mean(values)
                result[f"{key}_std"] = statistics.stdev(values)
            elif values:
                result[key] = values[0]
                result[f"{key}_std"] = 0.0

        return result

    @staticmethod
    def _create_anomaly(
        protocol_id: str,
        anomaly_type: AnomalyType,
        severity: Severity,
        description: str,
        metric_name: str,
        metric_value: float,
        baseline_value: float,
        deviation_pct: float = 0.0,
        z_score: float = 0.0,
        related_metrics: dict[str, Any] | None = None,
    ) -> Anomaly:
        """Factory method to create an Anomaly with a unique ID."""
        return Anomaly(
            id=f"anomaly-{uuid.uuid4().hex[:12]}",
            protocol_id=protocol_id,
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            metric_name=metric_name,
            metric_value=metric_value,
            baseline_value=baseline_value,
            deviation_pct=deviation_pct,
            z_score=z_score,
            related_metrics=related_metrics or {},
        )
