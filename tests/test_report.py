"""Tests for financial report generator."""

from src.utils.report import ReportConfig, _rate, generate_report


class TestRate:
    def test_strong_higher_better(self):
        assert _rate(50, 30, 10) == "Strong"

    def test_moderate_higher_better(self):
        assert _rate(20, 30, 10) == "Moderate"

    def test_weak_higher_better(self):
        assert _rate(5, 30, 10) == "Weak"

    def test_none(self):
        assert _rate(None, 30, 10) == "No data available"

    def test_chinese(self):
        assert _rate(50, 30, 10, "zh") == "强"
        assert _rate(5, 30, 10, "zh") == "弱"


class TestGenerateReport:
    def setup_method(self):
        self.income = {
            "Total Revenue": 130_000,
            "Gross Profit": 100_000,
            "Operating Income": 85_000,
            "Net Income": 72_000,
            "Research And Development": 12_000,
        }
        self.balance = {
            "Total Assets": 200_000,
            "Stockholders Equity": 80_000,
            "Total Debt": 10_000,
            "Current Assets": 50_000,
            "Current Liabilities": 20_000,
            "Cash And Cash Equivalents": 15_000,
        }
        self.cashflow = {
            "Free Cash Flow": 60_000,
            "Operating Cash Flow": 70_000,
            "Capital Expenditure": -10_000,
        }

    def test_basic_english(self):
        md = generate_report("NVIDIA", "NVDA", self.income, self.balance, self.cashflow)
        assert "NVIDIA" in md
        assert "NVDA" in md
        assert "Profitability" in md
        assert "Liquidity" in md
        assert "Leverage" in md
        assert "Cash Flow" in md

    def test_chinese(self):
        config = ReportConfig(language="zh")
        md = generate_report("英伟达", "NVDA", self.income, self.balance, self.cashflow, config=config)
        assert "英伟达" in md
        assert "盈利能力" in md
        assert "流动性" in md
        assert "杠杆" in md

    def test_with_trends(self):
        historical = {
            "2022": {"Total Revenue": 26_000, "Net Income": 9_700},
            "2023": {"Total Revenue": 27_000, "Net Income": 4_300},
            "2024": {"Total Revenue": 61_000, "Net Income": 29_700},
            "2025": {"Total Revenue": 130_000, "Net Income": 72_000},
        }
        md = generate_report("NVIDIA", "NVDA", self.income, self.balance, self.cashflow, historical=historical)
        assert "Trend" in md or "↑" in md

    def test_risk_detection(self):
        weak_income = {"Total Revenue": 100_000, "Gross Profit": 10_000, "Net Income": -5_000}
        weak_balance = {
            "Total Assets": 100_000,
            "Stockholders Equity": 20_000,
            "Total Debt": 60_000,
            "Current Assets": 10_000,
            "Current Liabilities": 15_000,
        }
        md = generate_report("WeakCo", "WEAK", weak_income, weak_balance)
        assert "⚠️" in md

    def test_no_risk(self):
        md = generate_report("NVIDIA", "NVDA", self.income, self.balance, self.cashflow)
        assert "✅" in md

    def test_empty_data(self):
        md = generate_report("Empty", "EMPT")
        assert "Empty" in md
        assert "N/A" in md

    def test_no_ratios(self):
        config = ReportConfig(include_ratios=False)
        md = generate_report("NVIDIA", "NVDA", self.income, config=config)
        assert "Profitability" not in md

    def test_no_trends(self):
        config = ReportConfig(include_trends=False)
        md = generate_report("NVIDIA", "NVDA", self.income, config=config)
        assert "Trend" not in md

    def test_summary_good(self):
        md = generate_report("NVIDIA", "NVDA", self.income, self.balance, self.cashflow)
        assert "strong" in md.lower() or "good" in md.lower() or "solid" in md.lower()

    def test_summary_chinese(self):
        config = ReportConfig(language="zh")
        md = generate_report("英伟达", "NVDA", self.income, self.balance, self.cashflow, config=config)
        assert "良好" in md or "强劲" in md or "中等" in md
