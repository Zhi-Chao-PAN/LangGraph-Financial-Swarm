"""
Tests for eval_utils: scale-aware number extraction, entity matching,
answer checking, nDCG@k, and MRR computation.

Covers improvements: 4 (numeric hardening) and 7 (IR metrics).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.eval_utils import (
    approx_match,
    check_answer,
    compute_dcg,
    compute_mrr,
    compute_ndcg,
    extract_entities,
    extract_numbers,
)


# ── Improvement 4: Numeric Extraction Hardening ────────────────────


class TestExtractNumbers:
    def test_plain_integer(self):
        assert 130497 in extract_numbers("Revenue was 130,497 million")

    def test_dollar_sign(self):
        nums = extract_numbers("$130,497M in revenue")
        assert any(abs(n - 130497) < 1 for n in nums)

    def test_percentage(self):
        nums = extract_numbers("margin was 77.6%")
        assert any(abs(n - 77.6) < 0.01 for n in nums)

    def test_bracket_negative(self):
        nums = extract_numbers("loss was (2,722)")
        assert any(n < 0 for n in nums)
        assert any(abs(n + 2722) < 1 for n in nums)

    def test_decimal(self):
        nums = extract_numbers("EPS was 2.94")
        assert any(abs(n - 2.94) < 0.01 for n in nums)

    def test_negative_number(self):
        nums = extract_numbers("change was -12.5")
        assert any(n < 0 for n in nums)

    def test_empty(self):
        assert extract_numbers("") == []

    def test_no_numbers(self):
        assert extract_numbers("revenue increased significantly") == []

    def test_multiple_numbers(self):
        nums = extract_numbers("from $15,005M to $115,199M")
        assert len(nums) >= 2


class TestApproxMatch:
    def test_exact(self):
        assert approx_match(100.0, 100.0)

    def test_within_tolerance(self):
        assert approx_match(104.0, 100.0, tol=0.05)

    def test_outside_tolerance(self):
        assert not approx_match(110.0, 100.0, tol=0.05)

    def test_zero_base(self):
        assert approx_match(0.005, 0)
        assert not approx_match(0.1, 0)


class TestExtractEntities:
    def test_company_names(self):
        entities = extract_entities("NVIDIA revenue was $130B")
        assert "nvidia" in entities

    def test_fiscal_year(self):
        entities = extract_entities("in FY2025")
        assert "fy2025" in entities

    def test_growth_terms(self):
        entities = extract_entities("revenue increased by 20%")
        assert "growth" in entities

    def test_decline_terms(self):
        entities = extract_entities("profits decreased by 10%")
        assert "decline" in entities

    def test_segments(self):
        entities = extract_entities("Data Center segment grew")
        assert "data center" in entities


class TestCheckAnswer:
    def test_exact_match(self):
        result = check_answer(
            "NVIDIA revenue was $130,497M",
            "$130,497M",
        )
        assert result["correct"]
        assert result["number_recall"] == 1.0

    def test_partial_match(self):
        result = check_answer(
            "Revenue was approximately $130,497M with growth",
            "Revenue was $130,497M, up 114% YoY",
        )
        # Numbers should match since 130497 is exact
        assert result["correct"]
        assert result["number_recall"] > 0

    def test_no_match(self):
        result = check_answer(
            "I don't have that information",
            "$130,497M",
        )
        assert not result["correct"]
        assert result["number_recall"] == 0.0

    def test_hallucination_detection(self):
        result = check_answer(
            "The quantum computing sector saw massive improvements in coherence times which enabled breakthroughs in drug discovery and materials science applications worldwide",
            "Revenue was $130,497M for Data Center in FY2025",
        )
        assert result["hallucination"]


# ── Improvement 7: IR Metrics ──────────────────────────────────────


class TestDCG:
    def test_perfect_ranking(self):
        # All relevant at top
        dcg = compute_dcg([1.0, 1.0, 1.0], k=3)
        assert dcg > 0

    def test_empty(self):
        assert compute_dcg([]) == 0

    def test_k_cutoff(self):
        dcg_full = compute_dcg([1.0, 0.0, 1.0])
        dcg_k1 = compute_dcg([1.0, 0.0, 1.0], k=1)
        assert dcg_k1 < dcg_full


class TestNDCG:
    def test_perfect(self):
        assert compute_ndcg([1.0, 1.0, 1.0], k=3) == 1.0

    def test_worst(self):
        # Relevant doc at the end
        ndcg = compute_ndcg([0.0, 0.0, 1.0], k=3)
        assert ndcg < 1.0
        assert ndcg > 0

    def test_empty(self):
        assert compute_ndcg([], k=5) == 0.0

    def test_all_irrelevant(self):
        assert compute_ndcg([0.0, 0.0, 0.0], k=3) == 0.0


class TestMRR:
    def test_all_first(self):
        assert compute_mrr([[1.0, 0.0], [1.0, 0.0]]) == 1.0

    def test_all_second(self):
        mrr = compute_mrr([[0.0, 1.0], [0.0, 1.0]])
        assert abs(mrr - 0.5) < 0.01

    def test_empty(self):
        assert compute_mrr([]) == 0.0

    def test_no_relevant(self):
        assert compute_mrr([[0.0, 0.0]]) == 0.0

    def test_mixed(self):
        mrr = compute_mrr([[1.0, 0.0], [0.0, 1.0]])
        assert abs(mrr - 0.75) < 0.01  # (1 + 0.5) / 2
