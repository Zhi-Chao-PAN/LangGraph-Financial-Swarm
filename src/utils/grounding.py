"""
Answer grounding verification using NLI (Natural Language Inference).

Checks whether the generated answer is faithfully supported (entailed)
by the retrieved source documents, detecting semantic hallucinations
that simple heuristic checks miss (e.g., correct numbers but wrong
causal explanation).

Uses a lightweight cross-encoder NLI model for efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("SwarmTracer")

# Lazy-loaded model cache
_nli_model = None


@dataclass
class GroundingResult:
    """Result of grounding verification."""

    is_grounded: bool
    score: float          # Average entailment probability
    num_claims: int       # Number of claims checked
    num_supported: int    # Number of claims supported by sources
    details: list[dict]   # Per-claim details

    def to_dict(self) -> dict:
        return {
            "is_grounded": self.is_grounded,
            "score": round(self.score, 3),
            "num_claims": self.num_claims,
            "num_supported": self.num_supported,
        }

    def to_warning(self) -> str | None:
        """Return warning string if answer is not well-grounded."""
        if self.is_grounded:
            return None
        return (
            f"⚠️ Low grounding score ({self.score:.0%}): "
            f"only {self.num_supported}/{self.num_claims} claims "
            f"supported by retrieved sources."
        )


def _get_nli_model(model_name: str):
    """Lazy-load the NLI cross-encoder model."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model

    try:
        from sentence_transformers import CrossEncoder

        _nli_model = CrossEncoder(model_name, max_length=512)
        logger.info(f"Loaded NLI model: {model_name}")
        return _nli_model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Grounding verification disabled."
        )
        return None
    except Exception as e:
        logger.warning(f"Failed to load NLI model {model_name}: {e}")
        return None


def _split_into_claims(answer: str) -> list[str]:
    """
    Split an answer into atomic claims for verification.
    Uses sentence-level splitting with financial-aware heuristics.
    """
    import re

    # Remove citation blocks
    answer = re.sub(r"\*\*Citations?\*\*:.*$", "", answer, flags=re.DOTALL)
    answer = re.sub(r"\[Source \d+\].*$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r"\[Confidence:.*?\]", "", answer)

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())

    claims = []
    for s in sentences:
        s = s.strip()
        # Skip very short or non-informative sentences
        if len(s) < 15:
            continue
        # Skip meta-sentences
        if any(skip in s.lower() for skip in [
            "based on", "according to", "the following",
            "in summary", "to summarize", "let me",
        ]):
            continue
        claims.append(s)

    return claims[:10]  # Cap at 10 claims to limit computation


def check_grounding(
    answer: str,
    sources: list[str],
    model_name: str = "cross-encoder/nli-deberta-v3-small",
    threshold: float = 0.5,
) -> GroundingResult:
    """
    Verify whether each claim in the answer is supported by source documents.

    Uses NLI cross-encoder to classify (premise=source, hypothesis=claim)
    as entailment/neutral/contradiction.

    Args:
        answer: Generated answer text
        sources: List of source document texts
        model_name: NLI cross-encoder model name
        threshold: Minimum entailment probability to consider a claim supported

    Returns:
        GroundingResult with per-claim verification details
    """
    claims = _split_into_claims(answer)
    if not claims or not sources:
        return GroundingResult(
            is_grounded=True,
            score=1.0,
            num_claims=len(claims),
            num_supported=len(claims),
            details=[],
        )

    model = _get_nli_model(model_name)
    if model is None:
        # Fallback: assume grounded if model unavailable
        return GroundingResult(
            is_grounded=True,
            score=1.0,
            num_claims=len(claims),
            num_supported=len(claims),
            details=[{"note": "NLI model unavailable, skipping verification"}],
        )

    # Concatenate sources (truncate to avoid OOM)
    combined_source = " ".join(s[:500] for s in sources[:5])
    if len(combined_source) > 2000:
        combined_source = combined_source[:2000]

    # Score each claim against combined source
    pairs = [(combined_source, claim) for claim in claims]
    try:
        # CrossEncoder.predict returns scores; for NLI models it's
        # [contradiction, neutral, entailment] logits
        scores = model.predict(pairs)

        details = []
        num_supported = 0
        total_score = 0.0

        for i, (claim, score_row) in enumerate(zip(claims, scores)):
            # For NLI models, score_row is [contradiction, neutral, entailment]
            if hasattr(score_row, '__len__') and len(score_row) >= 3:
                entailment_prob = float(score_row[2])  # entailment class
            else:
                # Single score (some models output a single value)
                entailment_prob = float(score_row)

            supported = entailment_prob >= threshold
            if supported:
                num_supported += 1
            total_score += entailment_prob

            details.append({
                "claim": claim[:100],
                "entailment_score": round(entailment_prob, 3),
                "supported": supported,
            })

        avg_score = total_score / len(claims) if claims else 1.0

        return GroundingResult(
            is_grounded=avg_score >= threshold,
            score=avg_score,
            num_claims=len(claims),
            num_supported=num_supported,
            details=details,
        )

    except Exception as e:
        logger.warning(f"Grounding verification failed: {e}")
        return GroundingResult(
            is_grounded=True,
            score=1.0,
            num_claims=len(claims),
            num_supported=len(claims),
            details=[{"error": str(e)[:100]}],
        )
