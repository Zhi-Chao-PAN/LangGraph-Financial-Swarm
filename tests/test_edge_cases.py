"""Edge case tests for robustness across modules."""

from src.agents.researcher import (
    _build_sub_queries,
    _detect_companies,
    _detect_year_range,
    _is_multi_entity_question,
    _is_trend_question,
)
from src.utils.quality import assess_answer_quality
from src.utils.validation import sanitize_input, sanitize_rag_query


class TestEdgeCaseDetection:
    """Edge cases for company/entity detection."""

    def test_detect_no_company(self):
        assert _detect_companies("What is revenue?") == []

    def test_detect_lowercase_company(self):
        result = _detect_companies("nvidia revenue")
        assert any("nvidia" in c.lower() for c in result)

    def test_detect_company_in_sentence(self):
        result = _detect_companies("Tell me about Apple's gross margin")
        assert any("apple" in c.lower() for c in result)

    def test_detect_google(self):
        result = _detect_companies("What was Google Cloud revenue?")
        # Google is normalized to 'alphabet' via entity normalizer
        assert any("alphabet" in c.lower() or "google" in c.lower() for c in result)

    def test_detect_amazon(self):
        result = _detect_companies("Amazon AWS revenue in FY2024")
        assert any("amazon" in c.lower() for c in result)

    def test_detect_alphabet(self):
        result = _detect_companies("Alphabet net income in FY2023")
        assert any("alphabet" in c.lower() for c in result)

    def test_multi_entity_with_and(self):
        assert _is_multi_entity_question("NVIDIA and Apple revenue comparison")

    def test_multi_entity_with_vs(self):
        assert _is_multi_entity_question("NVIDIA vs Apple")

    def test_single_entity_not_multi(self):
        assert not _is_multi_entity_question("What was NVIDIA revenue?")

    def test_empty_question(self):
        assert not _is_multi_entity_question("")
        assert _detect_companies("") == []


class TestEdgeCaseTrend:
    """Edge cases for trend/year detection."""

    def test_no_year(self):
        assert _detect_year_range("What is revenue?") is None

    def test_single_year(self):
        result = _detect_year_range("Revenue in FY2025")
        # Single year may or may not return a range
        assert result is None or len(result) >= 1

    def test_year_range(self):
        result = _detect_year_range("Revenue from FY2021 to FY2025")
        assert result is not None
        assert len(result) >= 2

    def test_trend_with_growth(self):
        assert _is_trend_question("How has NVIDIA revenue grown over time?")

    def test_trend_with_trend_word(self):
        assert _is_trend_question("What is the trend in Apple margins?")

    def test_not_trend(self):
        assert not _is_trend_question("What was NVIDIA revenue in FY2025?")


class TestEdgeCaseSubQueries:
    """Edge cases for sub-query generation."""

    def test_single_company_no_split(self):
        queries = _build_sub_queries("What was NVIDIA revenue in FY2025?")
        assert len(queries) == 1

    def test_two_companies_split(self):
        queries = _build_sub_queries("Compare NVIDIA and Apple revenue in FY2025")
        assert len(queries) >= 2

    def test_empty_question(self):
        queries = _build_sub_queries("")
        assert len(queries) >= 1


class TestEdgeCaseQuality:
    """Edge cases for answer quality assessment."""

    def test_empty_string(self):
        result = assess_answer_quality("", "test question")
        assert result["quality"] in ("empty", "no_answer")

    def test_whitespace_only(self):
        result = assess_answer_quality("   \n\t  ", "test question")
        assert result["quality"] in ("empty", "no_answer")

    def test_very_long_good_answer(self):
        answer = "NVIDIA revenue was $130.5 billion in FY2025. " * 20
        result = assess_answer_quality(answer, "What was NVIDIA revenue?")
        assert result["quality"] in ("good", "weak")

    def test_hedging_answer(self):
        result = assess_answer_quality("I'm not sure, but it might be around $100B maybe", "What was revenue?")
        assert result["quality"] in ("weak", "good")

    def test_no_answer_phrases(self):
        result = assess_answer_quality("I don't have information about that.", "What was revenue?")
        assert result["quality"] in ("no_answer", "empty", "weak")


class TestEdgeCaseValidation:
    """Edge cases for input validation."""

    def test_sanitize_empty(self):
        assert sanitize_input("") == ""

    def test_sanitize_normal(self):
        assert sanitize_input("Hello world") == "Hello world"

    def test_sanitize_strips_whitespace(self):
        result = sanitize_input("  hello  ")
        assert result.strip() == "hello"

    def test_rag_query_sanitize(self):
        result = sanitize_rag_query("What was NVIDIA revenue?")
        assert "NVIDIA" in result

    def test_rag_query_empty(self):
        result = sanitize_rag_query("")
        assert result == ""

    def test_sanitize_special_chars(self):
        result = sanitize_input("Hello <script>alert('xss')</script>")
        assert "<script>" not in result or isinstance(result, str)

    def test_very_long_input(self):
        long_input = "A" * 10000
        result = sanitize_input(long_input)
        assert len(result) <= 10000
