"""Tests for peer comparison analysis."""

from src.utils.peer_compare import (
    PeerComparison,
    PeerMetrics,
    compare_peers,
    get_industry,
    identify_peers,
)


class TestIdentifyPeers:
    def test_nvidia(self):
        peers = identify_peers("NVDA")
        assert "NVDA" in peers
        assert "AMD" in peers
        assert len(peers) <= 5

    def test_apple(self):
        peers = identify_peers("AAPL")
        assert "AAPL" in peers

    def test_unknown(self):
        peers = identify_peers("UNKNOWN_TICKER")
        assert peers == []

    def test_max_peers(self):
        peers = identify_peers("NVDA", max_peers=3)
        assert len(peers) <= 3

    def test_case_insensitive(self):
        peers = identify_peers("nvda")
        assert "NVDA" in peers


class TestGetIndustry:
    def test_known(self):
        assert get_industry("NVDA") == "semiconductors"
        assert get_industry("AAPL") == "consumer_electronics"
        assert get_industry("MSFT") == "software"

    def test_unknown(self):
        assert get_industry("UNKNOWN") == "unknown"


class TestComparePeers:
    def test_basic(self):
        peer_data = {
            "NVDA": {
                "income": {"Total Revenue": 130_000, "Gross Profit": 100_000, "Net Income": 72_000, "Operating Income": 85_000},
                "balance": {
                    "Total Assets": 200_000,
                    "Stockholders Equity": 80_000,
                    "Total Debt": 10_000,
                    "Current Assets": 50_000,
                    "Current Liabilities": 20_000,
                },
                "cashflow": {"Free Cash Flow": 60_000},
            },
            "AMD": {
                "income": {"Total Revenue": 23_000, "Gross Profit": 11_000, "Net Income": 1_600, "Operating Income": 2_000},
                "balance": {
                    "Total Assets": 67_000,
                    "Stockholders Equity": 55_000,
                    "Total Debt": 2_000,
                    "Current Assets": 15_000,
                    "Current Liabilities": 8_000,
                },
                "cashflow": {"Free Cash Flow": 1_200},
            },
        }
        result = compare_peers("NVDA", peer_data)
        assert isinstance(result, PeerComparison)
        assert result.target_ticker == "NVDA"
        assert len(result.peers) == 2

    def test_to_markdown(self):
        peer_data = {
            "NVDA": {
                "income": {"Total Revenue": 130_000, "Gross Profit": 100_000, "Net Income": 72_000},
                "balance": {"Total Assets": 200_000, "Stockholders Equity": 80_000},
            },
        }
        result = compare_peers("NVDA", peer_data)
        md = result.to_markdown()
        assert "NVDA" in md
        assert "Gross Margin" in md

    def test_empty_peers(self):
        result = compare_peers("NVDA", {})
        md = result.to_markdown()
        assert "No peer data" in md

    def test_custom_ratios(self):
        peer_data = {
            "NVDA": {
                "income": {"Total Revenue": 130_000, "Net Income": 72_000},
            },
        }
        result = compare_peers("NVDA", peer_data, ratios=["Net Margin"])
        assert result.comparison_ratios == ["Net Margin"]


class TestPeerMetrics:
    def test_get_ratio(self):
        from src.utils.calculator import RatioResult

        metrics = PeerMetrics(
            ticker="NVDA",
            name="NVIDIA",
            ratios=[RatioResult("Gross Margin", 77.6, "profitability", "desc")],
        )
        r = metrics.get_ratio("Gross Margin")
        assert r is not None
        assert r.value == 77.6

    def test_get_ratio_missing(self):
        metrics = PeerMetrics(ticker="NVDA", name="NVIDIA", ratios=[])
        assert metrics.get_ratio("NonExistent") is None
