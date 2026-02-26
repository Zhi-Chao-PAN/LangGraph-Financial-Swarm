"""Tests for multi-language support."""

from src.utils.i18n import detect_language, localize_response, translate_query


class TestDetectLanguage:
    def test_english(self):
        assert detect_language("What was NVIDIA revenue in FY2025?") == "en"

    def test_chinese(self):
        assert detect_language("英伟达2025财年的营收是多少？") == "zh"

    def test_mixed_mostly_chinese(self):
        assert detect_language("NVIDIA的净利润是多少？") == "zh"

    def test_mixed_mostly_english(self):
        assert detect_language("What was 英伟达 revenue?") == "en"

    def test_empty(self):
        assert detect_language("") == "en"

    def test_numbers_only(self):
        assert detect_language("12345") == "en"


class TestTranslateQuery:
    def test_chinese_revenue(self):
        result = translate_query("英伟达的营收是多少？")
        assert "NVIDIA" in result
        assert "revenue" in result

    def test_chinese_compare(self):
        result = translate_query("对比苹果和微软的净利润")
        assert "compare" in result
        assert "Apple" in result
        assert "Microsoft" in result
        assert "net income" in result

    def test_chinese_margin(self):
        result = translate_query("英伟达的毛利率")
        assert "gross margin" in result

    def test_english_passthrough(self):
        query = "What was NVIDIA revenue?"
        assert translate_query(query) == query

    def test_chinese_trend(self):
        result = translate_query("英伟达营收的趋势")
        assert "NVIDIA" in result
        assert "trend" in result

    def test_chinese_fy(self):
        result = translate_query("2025财年")
        assert "FY" in result

    def test_chinese_balance_sheet(self):
        result = translate_query("苹果的总资产")
        assert "Apple" in result
        assert "total assets" in result

    def test_chinese_cashflow(self):
        result = translate_query("亚马逊的自由现金流")
        assert "Amazon" in result
        assert "free cash flow" in result


class TestLocalizeResponse:
    def test_english_passthrough(self):
        resp = "Revenue was $130B"
        assert localize_response(resp, "en") == resp

    def test_chinese_localization(self):
        resp = "Total Revenue was $130,497 million. Net Income was $72,880 million."
        result = localize_response(resp, "zh")
        assert "总收入" in result
        assert "净利润" in result

    def test_preserves_numbers(self):
        resp = "Gross Margin was 77.6%"
        result = localize_response(resp, "zh")
        assert "77.6%" in result
        assert "毛利率" in result

    def test_case_insensitive(self):
        resp = "The gross margin is 60%"
        result = localize_response(resp, "zh")
        assert "毛利率" in result
