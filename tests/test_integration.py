"""Integration tests across multiple modules."""

from src.utils.calculator import FinancialCalculator
from src.utils.confidence import score_confidence
from src.utils.i18n import detect_language, localize_response, translate_query
from src.utils.report import ReportConfig, generate_report
from src.utils.trend import analyze_trend


class TestI18nIntegration:
    """Test i18n works end-to-end with other modules."""

    def test_chinese_query_to_english(self):
        query = "英伟达2025财年的毛利率是多少？"
        assert detect_language(query) == "zh"
        translated = translate_query(query)
        assert "NVIDIA" in translated
        assert "gross margin" in translated

    def test_localize_report_terms(self):
        text = "Total Revenue was $130,497M. Net Income was $72,880M. Gross Margin was 77.6%."
        localized = localize_response(text, "zh")
        assert "总收入" in localized
        assert "净利润" in localized
        assert "毛利率" in localized
        assert "$130,497M" in localized  # Numbers preserved


class TestCalculatorWithTrend:
    """Test calculator ratios feed into trend analysis."""

    def test_margin_trend(self):
        years_data = {
            "2021": {"Total Revenue": 100, "Gross Profit": 60},
            "2022": {"Total Revenue": 120, "Gross Profit": 78},
            "2023": {"Total Revenue": 150, "Gross Profit": 105},
            "2024": {"Total Revenue": 200, "Gross Profit": 150},
        }
        margins = []
        years = []
        for year, data in sorted(years_data.items()):
            calc = FinancialCalculator(income=data)
            result = calc.gross_margin()
            if result is not None:
                margins.append(result.value)
                years.append(year)

        trend = analyze_trend("gross_margin", years, margins)
        assert trend.direction in ("up", "stable")
        assert len(trend.values) == 4


class TestConfidenceWithSources:
    """Test confidence scoring with realistic inputs."""

    def test_high_confidence_answer(self):
        answer = "NVIDIA's total revenue in FY2025 was $130,497 million, representing a 114% year-over-year increase from $60,922 million in FY2024."
        result = score_confidence(
            answer=answer,
            query="What was NVIDIA total revenue in FY2025?",
            source_nodes=[
                "NVIDIA FY2025 Annual Report: Total Revenue $130,497M",
                "Income Statement: Revenue grew 114% YoY",
            ],
            retrieval_scores=[0.95, 0.88],
        )
        assert result.level == "high"
        assert len(result.sources) == 2
        assert result.sources[0].company == "NVIDIA"

    def test_low_confidence_hedging(self):
        result = score_confidence(
            answer="I'm not sure, but it might be around $100 billion maybe.",
            query="What was Apple revenue?",
            retrieval_scores=[0.3],
        )
        assert result.level != "high"


class TestReportWithCalculator:
    """Test report generation uses calculator correctly."""

    def test_report_with_full_data(self):
        income = {
            "Total Revenue": 130_497,
            "Net Income": 72_880,
            "Gross Profit": 100_000,
            "Operating Income": 90_000,
        }
        balance = {
            "Total Assets": 200_000,
            "Total Liabilities Net Minority Interest": 50_000,
            "Stockholders Equity": 150_000,
            "Current Assets": 80_000,
            "Current Liabilities": 30_000,
        }
        cashflow = {"Free Cash Flow": 60_000}

        report = generate_report(
            company_name="NVIDIA",
            ticker="NVDA",
            income=income,
            balance=balance,
            cashflow=cashflow,
            config=ReportConfig(language="en"),
        )
        assert "NVIDIA" in report
        assert "NVDA" in report

    def test_report_chinese(self):
        income = {"Total Revenue": 130_497, "Net Income": 72_880}
        report = generate_report(
            company_name="NVIDIA",
            ticker="NVDA",
            income=income,
            balance={},
            cashflow={},
            config=ReportConfig(language="zh"),
        )
        assert "NVIDIA" in report
