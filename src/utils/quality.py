"""
Answer quality assessment for the Researcher agent.
Detects low-quality, empty, or potentially hallucinated answers.
"""

import re
from typing import Any

from src.utils.robustness import log_agent_action

# Patterns indicating the model couldn't find an answer
NO_ANSWER_PATTERNS = [
    "does not include",
    "not found",
    "no information",
    "cannot find",
    "not available",
    "not provided",
    "i don't have",
    "i cannot",
    "unable to find",
    "no data",
    "not mentioned",
]

# Hedging patterns that reduce confidence
HEDGING_PATTERNS = [
    "approximately",
    "roughly",
    "around",
    "about",
    "estimated",
    "i think",
    "it seems",
    "possibly",
    "might be",
    "could be",
]

# Financial number pattern
_NUMBER_PATTERN = re.compile(r"[\$]?\d[\d,]*\.?\d*[%BMK]?")


def assess_answer_quality(answer: str, question: str) -> dict[str, Any]:
    """
    Assess the quality of a generated answer.

    Returns:
        {
            "quality": "good" | "weak" | "empty" | "no_answer",
            "has_numbers": bool,
            "hedging_count": int,
            "answer_length": int,
            "issues": list[str],
        }
    """
    issues = []
    answer_lower = answer.lower()

    # Empty or very short
    if not answer.strip():
        return {
            "quality": "empty",
            "has_numbers": False,
            "hedging_count": 0,
            "answer_length": 0,
            "issues": ["Answer is empty"],
        }

    if len(answer.strip()) < 20:
        issues.append("Answer is very short (<20 chars)")

    # No-answer detection
    no_answer_hits = sum(1 for p in NO_ANSWER_PATTERNS if p in answer_lower)
    if no_answer_hits >= 2:
        return {
            "quality": "no_answer",
            "has_numbers": False,
            "hedging_count": 0,
            "answer_length": len(answer),
            "issues": ["Model could not find the answer"],
        }

    # Number presence (financial answers should have numbers)
    has_numbers = bool(_NUMBER_PATTERN.search(answer))
    if not has_numbers:
        # Check if question expects numbers
        number_expecting = any(w in question.lower() for w in ["revenue", "income", "margin", "growth", "ratio", "how much", "calculate", "what was"])
        if number_expecting:
            issues.append("Answer lacks numbers for a quantitative question")

    # Hedging detection
    hedging_count = sum(1 for p in HEDGING_PATTERNS if p in answer_lower)
    if hedging_count >= 3:
        issues.append(f"High hedging ({hedging_count} hedging phrases)")

    # Repetition detection
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if s.strip()]
    if len(sentences) >= 2:
        unique_sentences = {s.lower() for s in sentences}
        if len(unique_sentences) < len(sentences) * 0.7:
            issues.append("Answer contains repetitive content")

    # Quality classification
    if issues:
        quality = "weak"
    else:
        quality = "good"

    result = {
        "quality": quality,
        "has_numbers": has_numbers,
        "hedging_count": hedging_count,
        "answer_length": len(answer),
        "issues": issues,
    }

    if quality != "good":
        log_agent_action("QualityCheck", quality, f"Issues: {issues}")

    return result
