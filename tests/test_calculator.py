"""Tests for financial ratio calculator."""

from src.utils.calculator import FinancialCalculator, RatioResult, _pct, _safe_div


class TestSafeDiv:
    def test_normal(self):
        assert _safe_div(10, 2) == 5.0

    def test_zero_divisor(self):
        assert _safe_div(10, 0) is None

    def test_none_numerator(self):
        assert _safe_div(None, 2) is None

    def test_none_denominator(self):
        assert _safe_div(10, None) is None


class TestPct:
    def test_normal(self):
        assert _pct(25, 100) == 25.0

    def test_none(self):
        assert _pct(None, 100) is None


class TestRatioResult:
    def test_formatted_pct(self):
        r = RatioResult("Test", 25.5, "test", "desc", "%")
        assert r.formatted() == "25.5%"

    def test_formatted_x(self):
        r = RatioResult("Test", 1.5, "test", "desc", "x")
        assert r.formatted() == "1.50x"

    def test_formatted_dollar(self):
        r = RatioResult("Test", 50000, "test", "desc", "$")
        assert r.formatted() == "$50,000"

    def test_formatted_none(self):
        r = RatioResult("Test", None, "test", "desc")
        assert r.formatted() == "N/A"


class TestProfitabilityRatios:
    def setup_method(self):
        self.calc = FinancialCalculator(
            income={
                "Total Revenue": 100_000,
                "Gross Profit": 60_000,
                "Operating Income": 30_000,
                "Net Income": 20_000,
                "Research And Development": 15_000,
            },
            balance={
                "Total Assets": 200_000,
                "Stockholders Equity": 80_000,
            },
        )

    def test_gross_margin(self):
        r = self.calc.gross_margin()
        assert r.value == 60.0
        assert r.category == "profitability"

    def test_operating_margin(self):
        r = self.calc.operating_margin()
        assert r.value == 30.0

    def test_net_margin(self):
        r = self.calc.net_margin()
        assert r.value == 20.0

    def test_roe(self):
        r = self.calc.roe()
        assert r.value == 25.0

    def test_roa(self):
        r = self.calc.roa()
        assert r.value == 10.0

    def test_rd_ratio(self):
        r = self.calc.rd_ratio()
        assert r.value == 15.0


class TestLiquidityRatios:
    def setup_method(self):
        self.calc = FinancialCalculator(
            balance={
                "Current Assets": 50_000,
                "Current Liabilities": 25_000,
                "Inventory": 10_000,
                "Cash And Cash Equivalents": 15_000,
            },
        )

    def test_current_ratio(self):
        r = self.calc.current_ratio()
        assert r.value == 2.0
        assert r.unit == "x"

    def test_quick_ratio(self):
        r = self.calc.quick_ratio()
        assert r.value == 1.6

    def test_cash_ratio(self):
        r = self.calc.cash_ratio()
        assert r.value == 0.6


class TestLeverageRatios:
    def setup_method(self):
        self.calc = FinancialCalculator(
            balance={
                "Total Debt": 40_000,
                "Stockholders Equity": 80_000,
                "Total Assets": 200_000,
            },
        )

    def test_debt_to_equity(self):
        r = self.calc.debt_to_equity()
        assert r.value == 0.5

    def test_debt_to_assets(self):
        r = self.calc.debt_to_assets()
        assert r.value == 20.0

    def test_equity_ratio(self):
        r = self.calc.equity_ratio()
        assert r.value == 40.0


class TestCashFlowRatios:
    def setup_method(self):
        self.calc = FinancialCalculator(
            income={"Total Revenue": 100_000},
            cashflow={
                "Free Cash Flow": 25_000,
                "Operating Cash Flow": 35_000,
                "Capital Expenditure": -10_000,
            },
        )

    def test_fcf_margin(self):
        r = self.calc.fcf_margin()
        assert r.value == 25.0

    def test_ocf_ratio(self):
        r = self.calc.ocf_ratio()
        assert r.value == 35.0

    def test_capex_ratio(self):
        r = self.calc.capex_ratio()
        assert r.value == 10.0


class TestComputeAll:
    def test_returns_all(self):
        calc = FinancialCalculator(
            income={"Total Revenue": 100_000, "Net Income": 20_000},
        )
        results = calc.compute_all()
        assert len(results) == 17
        assert all(isinstance(r, RatioResult) for r in results)

    def test_by_category(self):
        calc = FinancialCalculator(
            income={"Total Revenue": 100_000, "Net Income": 20_000},
        )
        profit = calc.compute_by_category("profitability")
        assert len(profit) == 6

    def test_summary(self):
        calc = FinancialCalculator(
            income={"Total Revenue": 100_000},
        )
        summary = calc.summary()
        assert "profitability" in summary
        assert "liquidity" in summary
        assert "leverage" in summary
        assert "cashflow" in summary

    def test_empty_data(self):
        calc = FinancialCalculator()
        results = calc.compute_all()
        assert all(r.value is None for r in results)
