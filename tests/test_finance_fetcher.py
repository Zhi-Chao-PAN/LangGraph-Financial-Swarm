"""Tests for Yahoo Finance data fetcher."""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.finance_fetcher import (
    _build_ratios_table,
    _build_table,
    _df_to_dict,
    _format_number,
    data_to_markdown,
)


class TestFormatNumber:
    def test_billions(self):
        assert _format_number(130_497_000_000) == "130,497"

    def test_millions(self):
        assert _format_number(5_500_000) == "5.5"

    def test_small(self):
        assert _format_number(1234) == "1,234"

    def test_negative(self):
        result = _format_number(-2_722_000_000)
        assert "-" in result


class TestDfToDict:
    def test_basic(self):
        df = pd.DataFrame(
            {"2024-01-01": [100, 200], "2023-01-01": [90, 180]},
            index=["Revenue", "Profit"],
        )
        df.columns = pd.to_datetime(df.columns)
        result = _df_to_dict(df)
        assert "2024" in result
        assert "2023" in result
        assert result["2024"]["Revenue"] == 100

    def test_empty(self):
        assert _df_to_dict(pd.DataFrame()) == {}

    def test_none(self):
        assert _df_to_dict(None) == {}


class TestBuildTable:
    def test_basic(self):
        data = {
            "2023": {"Total Revenue": 383_285_000_000, "Net Income": 96_995_000_000},
            "2024": {"Total Revenue": 391_035_000_000, "Net Income": 93_736_000_000},
        }
        result = _build_table(data, ["Total Revenue", "Net Income"])
        assert "Total Revenue" in result
        assert "Net Income" in result
        assert "FY2023" in result
        assert "FY2024" in result

    def test_empty_data(self):
        result = _build_table({}, ["Revenue"])
        assert "No data" in result

    def test_missing_metric(self):
        data = {"2024": {"Total Revenue": 100}}
        result = _build_table(data, ["Total Revenue", "NonExistent"])
        assert "Total Revenue" in result


class TestBuildRatiosTable:
    def test_basic(self):
        data = {
            "2023": {
                "Total Revenue": 383_285_000_000,
                "Gross Profit": 169_148_000_000,
                "Operating Income": 114_301_000_000,
                "Net Income": 96_995_000_000,
            },
            "2024": {
                "Total Revenue": 391_035_000_000,
                "Gross Profit": 180_683_000_000,
                "Operating Income": 123_216_000_000,
                "Net Income": 93_736_000_000,
            },
        }
        result = _build_ratios_table(data)
        assert "Gross Margin" in result
        assert "Operating Margin" in result
        assert "Net Margin" in result

    def test_empty(self):
        result = _build_ratios_table({})
        assert "No data" in result

    def test_yoy_growth(self):
        data = {
            "2023": {"Total Revenue": 100},
            "2024": {"Total Revenue": 120},
        }
        result = _build_ratios_table(data)
        assert "Revenue Growth" in result


class TestDataToMarkdown:
    def test_basic_structure(self):
        data = {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "info": {
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "United States",
                "currency": "USD",
                "market_cap": 3_000_000_000_000,
                "website": "https://apple.com",
            },
            "income_stmt": {"2024": {"Total Revenue": 391_035_000_000, "Net Income": 93_736_000_000}},
            "balance_sheet": {"2024": {"Total Assets": 364_980_000_000}},
            "cashflow": {"2024": {"Operating Cash Flow": 118_254_000_000}},
        }
        md = data_to_markdown(data)
        assert "Apple Inc." in md
        assert "Income Statement" in md
        assert "Balance Sheet" in md
        assert "Cash Flow" in md
        assert "AAPL" in md
        assert "$3.00T" in md

    def test_empty_data(self):
        data = {
            "ticker": "TEST",
            "name": "Test Corp",
            "info": {
                "sector": "N/A",
                "industry": "N/A",
                "country": "N/A",
                "currency": "USD",
                "market_cap": None,
                "website": "N/A",
            },
            "income_stmt": {},
            "balance_sheet": {},
            "cashflow": {},
        }
        md = data_to_markdown(data)
        assert "Test Corp" in md


class TestFetchWithMock:
    @patch("src.data.finance_fetcher.yf.Ticker")
    def test_fetch_company_data(self, mock_ticker_cls):
        from src.data.finance_fetcher import fetch_company_data

        mock_ticker = MagicMock()
        mock_ticker.info = {
            "longName": "Test Corp",
            "sector": "Tech",
            "industry": "Software",
            "country": "US",
            "currency": "USD",
            "marketCap": 1_000_000_000,
        }
        mock_ticker.income_stmt = pd.DataFrame(
            {"2024-01-01": [100_000_000]},
            index=["Total Revenue"],
        )
        mock_ticker.income_stmt.columns = pd.to_datetime(mock_ticker.income_stmt.columns)
        mock_ticker.balance_sheet = pd.DataFrame()
        mock_ticker.cashflow = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        result = fetch_company_data("TEST")
        assert result["name"] == "Test Corp"
        assert result["ticker"] == "TEST"
        assert "2024" in result["income_stmt"]
