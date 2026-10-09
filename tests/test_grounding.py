"""
Tests for NLI-based answer grounding verification.
Covers improvement 5.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.grounding import (
    GroundingResult,
    _split_into_claims,
    check_grounding,
)


class TestSplitIntoClaims:
    def test_basic_split(self):
        answer = "NVIDIA revenue was $130B. Net income was $72B."
        claims = _split_into_claims(answer)
        assert len(claims) == 2

    def test_filters_short(self):
        answer = "Yes. Revenue was $130B in FY2025."
        claims = _split_into_claims(answer)
        assert len(claims) == 1  # "Yes." too short

    def test_filters_meta_sentences(self):
        answer = "Based on the data, revenue was $130B. In summary, growth was strong."
        claims = _split_into_claims(answer)
        # Both "Based on..." and "In summary..." should be filtered
        assert len(claims) == 0

    def test_strips_citations(self):
        answer = "Revenue was $130B.\n**Citations**: Source 1, Source 2"
        claims = _split_into_claims(answer)
        assert len(claims) == 1
        assert "Citations" not in claims[0]

    def test_caps_at_10(self):
        answer = ". ".join(f"Claim number {i} with enough length to pass" for i in range(20)) + "."
        claims = _split_into_claims(answer)
        assert len(claims) <= 10


class TestGroundingResult:
    def test_grounded_no_warning(self):
        result = GroundingResult(
            is_grounded=True, score=0.8,
            num_claims=3, num_supported=3, details=[],
        )
        assert result.to_warning() is None

    def test_not_grounded_warning(self):
        result = GroundingResult(
            is_grounded=False, score=0.2,
            num_claims=5, num_supported=1, details=[],
        )
        warning = result.to_warning()
        assert warning is not None
        assert "1/5" in warning

    def test_to_dict(self):
        result = GroundingResult(
            is_grounded=True, score=0.85,
            num_claims=3, num_supported=3, details=[],
        )
        d = result.to_dict()
        assert d["is_grounded"] is True
        assert d["score"] == 0.85


class TestCheckGrounding:
    def test_empty_answer(self):
        result = check_grounding("", ["source text"])
        assert result.is_grounded

    def test_empty_sources(self):
        result = check_grounding("Some answer", [])
        assert result.is_grounded

    def test_fallback_when_model_unavailable(self):
        """Should gracefully fallback when NLI model is not installed."""
        result = check_grounding(
            "NVIDIA revenue was $130B in FY2025.",
            ["NVIDIA reported revenue of $130B."],
            model_name="nonexistent-model-12345",
        )
        # Should not raise, should return grounded (fallback)
        assert result.is_grounded
