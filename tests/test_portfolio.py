"""Tests for portfolio tracker."""

import json
import tempfile
from pathlib import Path

from src.core.portfolio import Holding, Portfolio, PortfolioSummary


class TestHolding:
    def test_basic(self):
        h = Holding(ticker="AAPL", shares=10, avg_cost=150.0)
        assert h.total_cost == 1500.0

    def test_to_dict(self):
        h = Holding(ticker="NVDA", shares=5, avg_cost=800.0)
        d = h.to_dict()
        assert d["ticker"] == "NVDA"
        assert d["total_cost"] == 4000.0


class TestPortfolioSummary:
    def test_to_dict(self):
        s = PortfolioSummary(
            total_cost=10000,
            total_value=12000,
            total_return=2000,
            return_pct=20.0,
            holdings_count=3,
            top_holding="NVDA",
            top_weight=50.0,
        )
        d = s.to_dict()
        assert d["return_pct"] == 20.0
        assert d["top_holding"] == "NVDA"

    def test_markdown_en(self):
        s = PortfolioSummary(
            total_cost=10000,
            total_value=12000,
            total_return=2000,
            return_pct=20.0,
            holdings_count=3,
            top_holding="NVDA",
            top_weight=50.0,
        )
        md = s.to_markdown("en")
        assert "Portfolio Summary" in md
        assert "NVDA" in md

    def test_markdown_zh(self):
        s = PortfolioSummary(
            total_cost=10000,
            total_value=12000,
            total_return=2000,
            return_pct=20.0,
            holdings_count=3,
            top_holding="NVDA",
            top_weight=50.0,
        )
        md = s.to_markdown("zh")
        assert "投资组合" in md
        assert "NVDA" in md


class TestPortfolio:
    def test_add(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)
        assert len(p.holdings) == 1
        assert p.get("AAPL").shares == 10

    def test_add_average_cost(self):
        p = Portfolio()
        p.add("AAPL", 10, 100.0)  # $1000
        p.add("AAPL", 10, 200.0)  # $2000
        h = p.get("AAPL")
        assert h.shares == 20
        assert h.avg_cost == 150.0  # $3000 / 20

    def test_add_uppercase(self):
        p = Portfolio()
        p.add("aapl", 10, 150.0)
        assert p.get("AAPL") is not None

    def test_remove_all(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)
        assert p.remove("AAPL")
        assert len(p.holdings) == 0

    def test_remove_partial(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)
        p.remove("AAPL", 5)
        assert p.get("AAPL").shares == 5

    def test_remove_nonexistent(self):
        p = Portfolio()
        assert not p.remove("AAPL")

    def test_tickers(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)
        p.add("NVDA", 5, 800.0)
        assert set(p.tickers) == {"AAPL", "NVDA"}

    def test_summary_no_prices(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)
        p.add("NVDA", 5, 800.0)
        s = p.summary()
        assert s.total_cost == 5500.0
        assert s.total_value == 5500.0
        assert s.total_return == 0.0
        assert s.holdings_count == 2

    def test_summary_with_prices(self):
        p = Portfolio()
        p.add("AAPL", 10, 150.0)  # cost $1500
        p.add("NVDA", 5, 800.0)  # cost $4000
        s = p.summary({"AAPL": 200.0, "NVDA": 1000.0})
        assert s.total_cost == 5500.0
        assert s.total_value == 7000.0  # 2000 + 5000
        assert s.total_return == 1500.0
        assert s.top_holding == "NVDA"

    def test_summary_empty(self):
        p = Portfolio()
        s = p.summary()
        assert s.holdings_count == 0
        assert s.total_cost == 0.0

    def test_allocation(self):
        p = Portfolio()
        p.add("AAPL", 10, 100.0)  # $1000
        p.add("NVDA", 10, 200.0)  # $2000
        alloc = p.allocation()
        assert len(alloc) == 2
        assert alloc[0]["ticker"] == "NVDA"  # Largest first
        assert alloc[0]["weight_pct"] > 60

    def test_allocation_empty(self):
        p = Portfolio()
        assert p.allocation() == []

    def test_save_and_load(self):
        p = Portfolio("Test Portfolio")
        p.add("AAPL", 10, 150.0)
        p.add("NVDA", 5, 800.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        p.save(path)

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["name"] == "Test Portfolio"
        assert len(data["holdings"]) == 2

        p2 = Portfolio()
        p2.load(path)
        assert p2.name == "Test Portfolio"
        assert len(p2.holdings) == 2
        assert p2.get("AAPL").shares == 10

        Path(path).unlink()

    def test_load_nonexistent(self):
        p = Portfolio()
        p.load("/nonexistent/path.json")
        assert len(p.holdings) == 0

    def test_return_pct(self):
        p = Portfolio()
        p.add("AAPL", 100, 100.0)  # cost $10000
        s = p.summary({"AAPL": 120.0})  # value $12000
        assert abs(s.return_pct - 20.0) < 0.01
