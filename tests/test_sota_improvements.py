"""
Tests for SOTA improvements across the swarm system.

Covers:
- Improvement 6: Weighted RRF fusion
- Improvement 8: Chain-of-Thought detection
- Improvement 9: Adaptive Top-K
- Improvement 10: Structured JSON routing
- Improvement 11: Answer deduplication
- Improvement 12: Financial entity normalizer
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Document

from src.agents.researcher import (
    ENTITY_ALIASES,
    _deduplicate_answers,
    _detect_companies,
    _needs_cot,
    normalize_entity,
)
from src.agents.supervisor import _parse_routing
from src.rag_adapter import (
    _add_context_overlap,
    _compute_adaptive_top_k,
    _detect_query_type,
    _reciprocal_rank_fusion,
)


# ── Improvement 6: Weighted RRF Fusion ─────────────────────────────


class TestWeightedRRF:
    def test_equal_weights(self):
        """Equal weights should behave like original RRF."""
        result = _reciprocal_rank_fusion(
            [[0, 1, 2], [0, 1, 2]], weights=[1.0, 1.0]
        )
        indices = [idx for idx, _ in result]
        assert indices[0] == 0

    def test_weighted_favor_first(self):
        """Higher weight on first ranking should boost its top doc."""
        result = _reciprocal_rank_fusion(
            [[0, 1, 2], [2, 1, 0]], weights=[2.0, 0.5]
        )
        scores = dict(result)
        # Doc 0 ranked #1 in first (weighted 2.0) and #3 in second (weighted 0.5)
        # Doc 2 ranked #3 in first (weighted 2.0) and #1 in second (weighted 0.5)
        # Doc 0 should score higher
        assert scores[0] > scores[2]

    def test_weighted_favor_second(self):
        """Higher weight on second ranking should boost its top doc."""
        result = _reciprocal_rank_fusion(
            [[0, 1, 2], [2, 1, 0]], weights=[0.5, 2.0]
        )
        scores = dict(result)
        assert scores[2] > scores[0]

    def test_default_weights(self):
        """No weights provided should default to uniform."""
        result = _reciprocal_rank_fusion([[0, 1], [1, 0]])
        assert len(result) == 2


# ── Improvement 3: Context Overlap ─────────────────────────────────


class TestContextOverlap:
    def test_adds_overlap(self):
        docs = [
            Document(text="Line 1\nLine 2\nLine 3\nLine 4", metadata={"company": "NVIDIA", "index": 0}),
            Document(text="New section content here", metadata={"company": "NVIDIA", "index": 1}),
        ]
        enhanced = _add_context_overlap(docs, overlap_lines=2)
        assert len(enhanced) == 2
        assert "[Context]" in enhanced[1].text
        assert "Line 3" in enhanced[1].text or "Line 4" in enhanced[1].text

    def test_no_overlap_different_companies(self):
        docs = [
            Document(text="NVIDIA data\nLine 2\nLine 3", metadata={"company": "NVIDIA", "index": 0}),
            Document(text="Apple data here", metadata={"company": "Apple", "index": 1}),
        ]
        enhanced = _add_context_overlap(docs, overlap_lines=2)
        assert "[Context]" not in enhanced[1].text

    def test_single_doc(self):
        docs = [Document(text="Only one doc", metadata={"company": "NVIDIA", "index": 0})]
        enhanced = _add_context_overlap(docs, overlap_lines=2)
        assert len(enhanced) == 1

    def test_zero_overlap(self):
        docs = [
            Document(text="Doc 1", metadata={"company": "NVIDIA", "index": 0}),
            Document(text="Doc 2", metadata={"company": "NVIDIA", "index": 1}),
        ]
        enhanced = _add_context_overlap(docs, overlap_lines=0)
        assert enhanced[1].text == "Doc 2"


# ── Improvement 9: Adaptive Top-K ──────────────────────────────────


class TestAdaptiveTopK:
    def test_simple_query(self):
        k = _compute_adaptive_top_k("What was NVIDIA revenue in FY2025?")
        assert k <= 5

    def test_multi_entity_query(self):
        k = _compute_adaptive_top_k(
            "Compare NVIDIA, Apple, and Microsoft revenue in FY2025"
        )
        assert k >= 10

    def test_trend_query(self):
        k = _compute_adaptive_top_k(
            "What was the CAGR of NVIDIA revenue from FY2023 to FY2025?"
        )
        assert k > 5

    def test_comparison_query(self):
        k = _compute_adaptive_top_k(
            "Which company had the highest revenue between NVIDIA and Apple?"
        )
        assert k > 5


class TestDetectQueryType:
    def test_numeric_query(self):
        assert _detect_query_type("What was NVIDIA revenue and income in FY2025?") == "numeric"

    def test_semantic_query(self):
        assert _detect_query_type("Describe NVIDIA's competitive advantage") == "semantic"


# ── Improvement 10: JSON Routing ───────────────────────────────────


class TestJSONRouting:
    def test_valid_json(self):
        content = '{"next": "Researcher", "reason": "factual question"}'
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Researcher"

    def test_json_with_markdown_fence(self):
        content = '```json\n{"next": "Quant", "reason": "chart requested"}\n```'
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Quant"

    def test_json_finish(self):
        content = '{"next": "FINISH", "reason": "already answered"}'
        assert _parse_routing(content, ["Researcher", "Quant"]) == "FINISH"

    def test_fallback_to_text(self):
        """When JSON fails, should fall back to text parsing."""
        content = "Researcher"
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Researcher"

    def test_fallback_next_pattern(self):
        content = "Next: Quant"
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Quant"

    def test_with_think_block(self):
        content = '<think>reasoning here</think>{"next": "Researcher", "reason": "data lookup"}'
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Researcher"

    def test_invalid_json_fallback(self):
        content = "I think we should route to the Researcher agent."
        result = _parse_routing(content, ["Researcher", "Quant"])
        assert result in ["Researcher", "Quant"]


# ── Improvement 12: Entity Normalizer ──────────────────────────────


class TestEntityNormalizer:
    def test_full_name(self):
        assert normalize_entity("NVIDIA") == "nvidia"

    def test_ticker(self):
        assert normalize_entity("NVDA") == "nvidia"

    def test_google_alphabet(self):
        assert normalize_entity("Google") == "alphabet"
        assert normalize_entity("GOOG") == "alphabet"
        assert normalize_entity("GOOGL") == "alphabet"

    def test_amazon(self):
        assert normalize_entity("AMZN") == "amazon"
        assert normalize_entity("Amazon") == "amazon"

    def test_unknown_entity(self):
        assert normalize_entity("Tesla") == "tesla"

    def test_alias_map_complete(self):
        # All canonical names should be values
        canonical = set(ENTITY_ALIASES.values())
        assert "nvidia" in canonical
        assert "apple" in canonical
        assert "microsoft" in canonical
        assert "alphabet" in canonical
        assert "amazon" in canonical


class TestDetectCompanies:
    def test_single_company(self):
        companies = _detect_companies("What was NVIDIA revenue?")
        assert "nvidia" in companies

    def test_multiple_companies(self):
        companies = _detect_companies("Compare Apple and Microsoft revenue")
        assert "apple" in companies
        assert "microsoft" in companies

    def test_ticker_detection(self):
        companies = _detect_companies("How did NVDA perform vs AAPL?")
        assert "nvidia" in companies
        assert "apple" in companies

    def test_google_alphabet_interchangeable(self):
        companies = _detect_companies("Google cloud revenue growth")
        assert "alphabet" in companies


# ── Improvement 8: Chain-of-Thought Detection ──────────────────────


class TestCoTDetection:
    def test_calculation_query(self):
        assert _needs_cot("Calculate the CAGR of NVIDIA revenue")

    def test_ratio_query(self):
        assert _needs_cot("What is the debt-to-equity ratio?")

    def test_percentage_query(self):
        assert _needs_cot("What percentage of revenue came from Data Center?")

    def test_comparison_query(self):
        assert _needs_cot("Compare NVIDIA and Apple gross margins")

    def test_simple_lookup(self):
        assert not _needs_cot("What was NVIDIA revenue in FY2025?")


# ── Improvement 11: Answer Deduplication ───────────────────────────


class TestAnswerDeduplication:
    def test_removes_exact_duplicates(self):
        queries = ["Q1", "Q2", "Q3"]
        answers = ["Revenue was $130B", "Revenue was $130B", "Margin was 75%"]
        dq, da = _deduplicate_answers(queries, answers)
        assert len(da) == 2
        assert "Margin was 75%" in da

    def test_no_duplicates(self):
        queries = ["Q1", "Q2"]
        answers = ["Revenue was $130B", "Margin was 75%"]
        dq, da = _deduplicate_answers(queries, answers)
        assert len(da) == 2

    def test_strips_citations_for_comparison(self):
        queries = ["Q1", "Q2"]
        answers = [
            "Revenue was $130B\n**Citations**: Source 1",
            "Revenue was $130B\n**Citations**: Source 2",
        ]
        dq, da = _deduplicate_answers(queries, answers)
        # After stripping citations, these should be considered duplicates
        assert len(da) == 1

    def test_preserves_order(self):
        queries = ["Q1", "Q2", "Q3"]
        answers = ["First", "Second", "First"]
        dq, da = _deduplicate_answers(queries, answers)
        assert da[0] == "First"
        assert da[1] == "Second"
