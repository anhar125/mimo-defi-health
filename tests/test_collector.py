"""Tests for CollectorAgent."""

from __future__ import annotations

import pytest
import httpx

from mimo_defi_health.agents.collector import CollectorAgent
from mimo_defi_health.models import ProtocolChain


@pytest.fixture
def mock_defillama_response() -> dict:
    """Mock DefiLlama API response."""
    return {
        "name": "Aave",
        "slug": "aave",
        "chainTvls": {
            "Ethereum": {"tvl": 8_500_000_000},
            "Arbitrum": {"tvl": 1_200_000_000},
        },
        "tvl": [
            {"date": 1700000000, "totalLiquidityUSD": 9_000_000_000},
            {"date": 1700086400, "totalLiquidityUSD": 8_800_000_000},
            {"date": 1700172800, "totalLiquidityUSD": 8_500_000_000},
        ],
        "apy": 4.25,
        "apyBorrow": 6.8,
        "utilization": {"overall": 0.78},
    }


class TestCollectorAgentInit:
    """Tests for CollectorAgent initialization."""

    def test_default_init(self) -> None:
        agent = CollectorAgent()
        assert agent._api_key == ""
        assert agent._timeout == 30.0
        assert agent._metrics_cache == {}

    def test_custom_init(self) -> None:
        agent = CollectorAgent(api_key="test-key", timeout=60.0)
        assert agent._api_key == "test-key"
        assert agent._timeout == 60.0


class TestCollectorAgentNormalization:
    """Tests for data normalization methods."""

    def test_normalize_protocol_data(self) -> None:
        agent = CollectorAgent()
        raw_data = {
            "name": "Test Protocol",
            "chainTvls": {"Ethereum": {"tvl": 1_000_000_000}},
            "tvl": [
                {"date": 1700000000, "totalLiquidityUSD": 1_100_000_000},
                {"date": 1700086400, "totalLiquidityUSD": 1_000_000_000},
            ],
            "apy": 5.0,
            "apyBorrow": 8.0,
        }

        metrics = agent._normalize_protocol_data("test-protocol", raw_data, None)
        assert metrics.protocol_name == "Test Protocol"
        assert metrics.tvl_total == 1_000_000_000
        assert metrics.supply_apy == 5.0
        assert metrics.borrow_apy == 8.0

    def test_resolve_chain(self) -> None:
        assert CollectorAgent._resolve_chain("ethereum") == ProtocolChain.ETHEREUM
        assert CollectorAgent._resolve_chain("arbitrum") == ProtocolChain.ARBITRUM
        assert CollectorAgent._resolve_chain("polygon") == ProtocolChain.POLYGON
        assert CollectorAgent._resolve_chain("unknown") == ProtocolChain.ETHEREUM

    def test_extract_chain_data_with_filter(self) -> None:
        agent = CollectorAgent()
        raw_data = {
            "chainTvls": {
                "Ethereum": {"tvl": 5_000_000_000},
                "Arbitrum": {"tvl": 1_000_000_000},
            }
        }
        result = agent._extract_chain_data(raw_data, [ProtocolChain.ARBITRUM])
        assert result["tvl"] == 1_000_000_000


class TestCollectorAgentAsync:
    """Tests for async operations (using mocked HTTP)."""

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        agent = CollectorAgent()
        async with agent as a:
            assert a._client is not None
        assert a._client is None

    @pytest.mark.asyncio
    async def test_collect_all_empty_list(self) -> None:
        agent = CollectorAgent()
        async with agent as a:
            result = await a.collect_all([])
            assert result == []

    @pytest.mark.asyncio
    async def test_collect_all_handles_failure(self) -> None:
        """Collector should handle individual protocol failures gracefully."""
        agent = CollectorAgent()
        async with agent as a:
            # Will fail to connect but shouldn't crash
            result = await a.collect_all(["nonexistent-protocol"])
            assert result == []
