"""Tests for answer quality assessment."""

from src.utils.quality import assess_answer_quality


class TestAnswerQuality:
    def test_good_answer(self):
        result = assess_answer_quality(
            "NVIDIA revenue in FY2025 was $130,497 million, a 114% increase from FY2024.",
            "What was NVIDIA revenue in FY2025?",
        )
        assert result["quality"] == "good"
        assert result["has_numbers"]
        assert result["issues"] == []

    def test_empty_answer(self):
        result = assess_answer_quality("", "What was revenue?")
        assert result["quality"] == "empty"
        assert result["answer_length"] == 0

    def test_no_answer(self):
        result = assess_answer_quality(
            "The information is not available. I cannot find the data. No information provided.",
            "What was Apple revenue?",
        )
        assert result["quality"] == "no_answer"

    def test_weak_no_numbers(self):
        result = assess_answer_quality(
            "NVIDIA had strong revenue growth in the fiscal year due to data center demand.",
            "What was NVIDIA revenue in FY2025?",
        )
        assert result["quality"] == "weak"
        assert not result["has_numbers"]
        assert any("numbers" in i.lower() for i in result["issues"])

    def test_hedging_detection(self):
        result = assess_answer_quality(
            "Revenue was approximately around $130B, roughly estimated at about that level, possibly higher.",
            "What was revenue?",
        )
        assert result["hedging_count"] >= 3

    def test_short_answer(self):
        result = assess_answer_quality("Yes.", "Is NVIDIA profitable?")
        assert any("short" in i.lower() for i in result["issues"])

    def test_non_quantitative_question(self):
        """Non-quantitative questions shouldn't penalize for missing numbers."""
        result = assess_answer_quality(
            "NVIDIA's Data Center segment was the largest contributor to overall performance.",
            "Which segment performed best?",
        )
        assert result["quality"] == "good"

    def test_answer_with_citations(self):
        result = assess_answer_quality(
            "Revenue was $130,497M.\n\n**Citations**:\n[Source 1] Revenue table...",
            "What was revenue?",
        )
        assert result["quality"] == "good"
        assert result["has_numbers"]

    def test_repetitive_answer(self):
        result = assess_answer_quality(
            "Revenue was high. Revenue was high. Revenue was high. Revenue was high.",
            "What was revenue?",
        )
        assert any("repetitive" in i.lower() for i in result["issues"])
