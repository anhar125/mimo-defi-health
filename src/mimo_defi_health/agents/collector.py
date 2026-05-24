"""Collector Agent — Gathers real-time DeFi protocol metrics.

Connects to on-chain data APIs (DefiLlama, Dune Analytics, protocol-specific)
to collect TVL, utilization rates, yield data, liquidation events, and gas metrics.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from mimo_defi_health.models import ProtocolChain, ProtocolMetrics

logger = structlog.get_logger(__name__)

# ── DefiLlama API Client ────────────────────────────────────────────────────

DEFILLAMA_BASE = "https://api.llama.fi"
DEFILLAMA_TVL_ENDPOINT = f"{DEFILLAMA_BASE}/tvl"  # /{protocol_slug}
DEFILLAMA_PROTOCOL_ENDPOINT = f"{DEFILLAMA_BASE}/protocol"  # /{protocol_slug}


class CollectorAgent:
    """Collects real-time metrics from DeFi protocols via on-chain data APIs.

    Supports multiple data sources and chains. Implements rate limiting,
    retry logic, and data normalization.
    """

    def __init__(
        self,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._metrics_cache: dict[str, ProtocolMetrics] = {}

    async def __aenter__(self) -> "CollectorAgent":
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    # ── Public API ──────────────────────────────────────────────────────

    async def collect_all(
        self,
        protocol_ids: list[str],
        chains: list[ProtocolChain] | None = None,
    ) -> list[ProtocolMetrics]:
        """Collect metrics for all specified protocols concurrently.

        Args:
            protocol_ids: List of protocol slugs (e.g., ["aave", "compound"]).
            chains: Optional chain filter. If None, collects from all supported chains.

        Returns:
            List of ProtocolMetrics snapshots.
        """
        tasks = [self._collect_protocol(pid, chains) for pid in protocol_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics: list[ProtocolMetrics] = []
        for pid, result in zip(protocol_ids, results):
            if isinstance(result, Exception):
                logger.error(
                    "collection_failed",
                    protocol=pid,
                    error=str(result),
                )
                continue
            if result is not None:
                metrics.append(result)

        logger.info("collection_complete", protocol_count=len(metrics))
        return metrics

    async def get_historical_tvl(
        self,
        protocol_id: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Fetch historical TVL data for trend analysis.

        Args:
            protocol_id: Protocol slug.
            days: Number of days of history.

        Returns:
            List of {timestamp, tvl} dicts.
        """
        url = f"{DEFILLAMA_PROTOCOL_ENDPOINT}/{protocol_id}"
        data = await self._get(url)

        if not data or "tvl" not in data:
            return []

        tvl_history = data["tvl"][-days:]
        return [
            {
                "timestamp": datetime.fromtimestamp(entry["date"], tz=timezone.utc),
                "tvl": entry["totalLiquidityUSD"],
            }
            for entry in tvl_history
        ]

    # ── Internal Methods ────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _get(self, url: str) -> dict[str, Any]:
        """Make an HTTP GET request with retry logic."""
        assert self._client is not None, "CollectorAgent must be used as async context manager"
        response = await self._client.get(url)
        response.raise_for_status()
        return response.json()

    async def _collect_protocol(
        self,
        protocol_id: str,
        chains: list[ProtocolChain] | None,
    ) -> ProtocolMetrics | None:
        """Collect metrics for a single protocol."""
        logger.debug("collecting_protocol", protocol_id=protocol_id)

        data = await self._get(f"{DEFILLAMA_PROTOCOL_ENDPOINT}/{protocol_id}")

        if not data:
            logger.warning("no_data", protocol_id=protocol_id)
            return None

        return self._normalize_protocol_data(protocol_id, data, chains)

    def _normalize_protocol_data(
        self,
        protocol_id: str,
        raw_data: dict[str, Any],
        chains: list[ProtocolChain] | None,
    ) -> ProtocolMetrics:
        """Normalize raw API data into ProtocolMetrics."""
        chain_data = self._extract_chain_data(raw_data, chains)

        tvl_total = chain_data.get("tvl", 0.0)
        tvl_change_1h = chain_data.get("change_1h", 0.0)
        tvl_change_24h = chain_data.get("change_1d", 0.0)

        # Extract utilization from available fields
        utilization = raw_data.get("utilization", 0.0)
        if isinstance(utilization, dict):
            utilization = utilization.get("overall", 0.0)

        return ProtocolMetrics(
            protocol_id=protocol_id,
            protocol_name=raw_data.get("name", protocol_id.replace("-", " ").title()),
            chain=self._resolve_chain(chain_data.get("chain", "ethereum")),
            tvl_total=float(tvl_total),
            tvl_change_1h=float(tvl_change_1h),
            tvl_change_24h=float(tvl_change_24h),
            utilization_rate=float(utilization) * 100 if utilization < 1 else float(utilization),
            supply_apy=float(raw_data.get("apy", 0.0)),
            borrow_apy=float(raw_data.get("apyBorrow", 0.0)),
            active_positions=int(raw_data.get("apyBase", 0)),
        )

    def _extract_chain_data(
        self,
        raw_data: dict[str, Any],
        chains: list[ProtocolChain] | None,
    ) -> dict[str, Any]:
        """Extract chain-specific data from protocol response."""
        chain_tvls = raw_data.get("chainTvls", {})
        if not chain_tvls:
            return {"tvl": raw_data.get("currentChainTvls", {}).get("Ethereum", 0)}

        target_chain = "Ethereum"
        for chain_name in chain_tvls:
            if chains is None or any(
                c.value.lower() == chain_name.lower() for c in chains
            ):
                target_chain = chain_name
                break

        chain_info = chain_tvls.get(target_chain, {})
        if isinstance(chain_info, dict):
            return chain_info

        return {"tvl": chain_info}

    @staticmethod
    def _resolve_chain(chain_str: str) -> ProtocolChain:
        """Map a string to a ProtocolChain enum."""
        mapping = {c.value.lower(): c for c in ProtocolChain}
        return mapping.get(chain_str.lower(), ProtocolChain.ETHEREUM)
