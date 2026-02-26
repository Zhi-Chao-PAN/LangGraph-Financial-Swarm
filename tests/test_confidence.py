"""Tests for confidence scoring and source attribution."""

from src.utils.confidence import (
    ConfidenceResult,
    SourceAttribution,
    score_confidence,
)


class TestSourceAttribution:
    def test_basic(self):
        sa = SourceAttribution(
            source_id="src_1",
            content_snippet="NVIDIA revenue was $130B",
            relevance_score=0.9,
            company="NVIDIA",
            section="Income Statement",
        )
        d = sa.to_dict()
        assert d["source_id"] == "src_1"
        assert d["relevance"] == 0.9
        assert d["company"] == "NVIDIA"

    def test_long_snippet_truncated(self):
        sa = SourceAttribution(
            source_id="src_1",
            content_snippet="x" * 500,
            relevance_score=0.5,
        )
        assert len(sa.to_dict()["snippet"]) == 200


class TestConfidenceResult:
    def test_to_dict(self):
        cr = ConfidenceResult(score=0.85, level="high", reasons=["Good"])
        d = cr.to_dict()
        assert d["score"] == 0.85
        assert d["level"] == "high"

    def test_to_markdown_high(self):
        cr = ConfidenceResult(score=0.85, level="high", reasons=["Detailed"])
        md = cr.to_markdown()
        assert "🟢" in md
        assert "85%" in md

    def test_to_markdown_low(self):
        cr = ConfidenceResult(score=0.2, level="low")
        md = cr.to_markdown()
        assert "🔴" in md


class TestScoreConfidence:
    def test_good_answer(self):
        result = score_confidence(
            answer="NVIDIA's total revenue in FY2025 was $130,497 million, a 114% increase YoY.",
            query="What was NVIDIA revenue in FY2025?",
            retrieval_scores=[0.9, 0.85, 0.8],
        )
        assert result.level == "high"
        assert result.score >= 0.7

    def test_empty_answer(self):
        result = score_confidence(answer="", query="What was revenue?")
        assert result.level == "low"
        assert result.score < 0.4

    def test_hedging_answer(self):
        result = score_confidence(
            answer="I'm not sure, but it might be around $100B maybe.",
            query="What was revenue?",
        )
        assert "Hedging" in str(result.reasons)
        assert result.score < 0.7

    def test_numeric_evidence_boost(self):
        result_with = score_confidence(
            answer="Revenue was $130,497M and net income was $72,880M.",
            query="What was revenue?",
        )
        result_without = score_confidence(
            answer="Revenue increased significantly year over year.",
            query="What was revenue?",
        )
        assert result_with.score > result_without.score

    def test_source_attribution(self):
        result = score_confidence(
            answer="Revenue was $130B",
            query="NVIDIA revenue",
            source_nodes=["NVIDIA FY2025 revenue data", "Apple income statement"],
            retrieval_scores=[0.9, 0.6],
        )
        assert len(result.sources) == 2
        assert result.sources[0].source_id == "src_1"
        assert result.sources[0].company == "NVIDIA"

    def test_low_retrieval_scores(self):
        result = score_confidence(
            answer="Some answer here with details.",
            query="What was revenue?",
            retrieval_scores=[0.2, 0.1],
        )
        assert "Low source relevance" in str(result.reasons)

    def test_high_retrieval_scores(self):
        result = score_confidence(
            answer="Revenue was $130,497M in FY2025.",
            query="What was revenue?",
            retrieval_scores=[0.9, 0.85, 0.88],
        )
        assert "High source relevance" in str(result.reasons)

    def test_query_answer_alignment(self):
        result = score_confidence(
            answer="NVIDIA revenue in FY2025 was $130,497 million.",
            query="What was NVIDIA revenue in FY2025?",
        )
        assert "alignment" in str(result.reasons).lower()

    def test_score_clamped(self):
        # Very bad answer should not go below 0
        result = score_confidence(
            answer="",
            query="test",
            retrieval_scores=[0.1],
        )
        assert result.score >= 0.0

    def test_no_sources(self):
        result = score_confidence(
            answer="Revenue was $130B",
            query="revenue",
        )
        assert len(result.sources) == 0

    def test_dict_source_nodes(self):
        result = score_confidence(
            answer="Revenue was $130B",
            query="revenue",
            source_nodes=[{"text": "NVIDIA revenue data here"}],
            retrieval_scores=[0.8],
        )
        assert len(result.sources) == 1
        assert result.sources[0].company == "NVIDIA"

    def test_section_extraction(self):
        result = score_confidence(
            answer="Revenue was $130B",
            query="revenue",
            source_nodes=["# Income Statement\nNVIDIA revenue was $130B"],
        )
        assert result.sources[0].section == "Income Statement"
