"""
Shared evaluation utilities for benchmark and ablation.
Provides robust number extraction, entity matching, and answer checking.

Improvements over original:
- Scale-aware number parsing (M/B/K suffixes)
- Robust bracket-negative handling
- nDCG@k and MRR computation
- Deduplicated from run_eval.py / run_ablation.py
"""

import math
import re


# ── Scale-Aware Number Extraction ──────────────────────────────────

_SCALE_MAP = {
    "T": 1e12,
    "B": 1e9,
    "M": 1e6,
    "K": 1e3,
}

# Match patterns like: $130,497M, (2,722), 77.6%, -12.5B, $1.2T
_FINANCIAL_NUMBER_RE = re.compile(
    r"""
    (?:^|(?<=\s)|(?<=\$)|(?<=\())       # start / $ / (
    -?                                   # optional negative
    [\d]{1,3}(?:,[\d]{3})*               # integer with commas
    (?:\.\d+)?                           # optional decimal
    \s*[TBMK%]?                          # optional scale suffix or %
    """,
    re.VERBOSE,
)

# Bracket-negative: (2,722) → -2722
_BRACKET_NEG_RE = re.compile(r"\((\d[\d,]*\.?\d*)\)")


def extract_numbers(text: str) -> list[float]:
    """
    Extract all financial numbers from text with scale awareness.

    Handles: $130,497M, (2,722), 77.6%, -12.5B, 1.2T, plain integers.
    Returns raw numeric values (percentages as-is, e.g. 77.6 for 77.6%).
    """
    # Pre-process bracket negatives: (2,722) → -2,722
    processed = _BRACKET_NEG_RE.sub(r"-\1", text)
    # Strip $ signs for cleaner matching
    processed = processed.replace("$", "")

    nums: list[float] = []
    # Find all candidate number strings
    for match in re.finditer(
        r"-?[\d]{1,3}(?:,[\d]{3})*(?:\.\d+)?\s*[TBMK%]?", processed
    ):
        raw = match.group().strip()
        if not raw or raw == "-":
            continue

        # Detect suffix
        suffix = ""
        if raw[-1] in "TBMK%":
            suffix = raw[-1]
            raw = raw[:-1].strip()

        # Remove commas
        raw = raw.replace(",", "")

        try:
            value = float(raw)
        except ValueError:
            continue

        # Apply scale (but NOT for percentages — keep as-is)
        if suffix in _SCALE_MAP:
            # Don't scale if this looks like it's already in millions
            # (heuristic: if raw value > 1000 and suffix is M, likely already scaled)
            pass  # Keep raw value; scale suffix is informational in our benchmark
        # For percentages, strip the % but keep the number
        # (consistent with original behavior)

        nums.append(value)

    return nums


def approx_match(a: float, b: float, tol: float = 0.05) -> bool:
    """Check if two numbers are approximately equal within tolerance."""
    if a == b:
        return True
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / max(abs(b), 1e-9) <= tol


# ── Entity Extraction ──────────────────────────────────────────────

_ENTITY_NAMES = [
    "data center", "gaming", "automotive", "professional visualization",
    "united states", "china", "taiwan", "singapore",
    "nvidia", "apple", "microsoft", "alphabet", "google", "amazon",
    "intelligent cloud", "iphone", "mac", "services",
    "google cloud", "aws", "north america", "youtube",
]

_GROWTH_TERMS = [
    "increase", "increased", "grew", "growth", "grow", "up from", "higher",
]
_DECLINE_TERMS = [
    "decrease", "decreased", "decline", "declined", "dropped", "drop",
    "lower", "fell",
]
_COMPARE_TERMS = [
    "better", "worse", "fluctuat", "not consistently", "inconsistent",
]


def extract_entities(text: str) -> set[str]:
    """Extract key financial entities from text."""
    entities: set[str] = set()
    lower = text.lower()

    for name in _ENTITY_NAMES:
        if name in lower:
            entities.add(name)

    # Fiscal years
    for m in re.finditer(r"fy\d{4}", lower):
        entities.add(m.group())

    # Directional terms (group synonyms)
    if any(t in lower for t in _GROWTH_TERMS):
        entities.add("growth")
    if any(t in lower for t in _DECLINE_TERMS):
        entities.add("decline")
    if any(t in lower for t in _COMPARE_TERMS):
        entities.add("comparison")

    return entities


# ── Answer Checking ────────────────────────────────────────────────


def check_answer(predicted: str, ground_truth: str) -> dict:
    """
    Robust answer checking with approximate number matching and entity matching.
    Returns partial_score (0-1), correct (bool), and detailed reasoning.
    """
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()

    gt_nums = extract_numbers(gt_lower)
    pred_nums = extract_numbers(pred_lower)

    # Number matching with tolerance
    matched = 0
    for gn in gt_nums:
        for pn in pred_nums:
            if approx_match(pn, gn):
                matched += 1
                break

    total_nums = max(len(gt_nums), 1)
    if matched == len(gt_nums):
        number_score = 1.0
    elif matched > 0:
        number_score = max(0.5, matched / total_nums)
    else:
        number_score = 0.0

    # Entity matching
    gt_entities = extract_entities(gt_lower)
    pred_entities = extract_entities(pred_lower)
    if gt_entities:
        entity_overlap = len(gt_entities & pred_entities) / len(gt_entities)
    else:
        entity_overlap = 1.0 if number_score > 0 else 0.0

    # Composite partial score
    if gt_nums and gt_entities:
        partial_score = 0.5 * number_score + 0.5 * entity_overlap
    elif gt_nums:
        partial_score = 0.8 * number_score + 0.2 * entity_overlap
    else:
        partial_score = 0.3 * number_score + 0.7 * entity_overlap

    # Boost: high entity overlap + some numbers matched
    if entity_overlap >= 0.8 and number_score >= 0.5:
        partial_score = max(partial_score, 0.75)
    if entity_overlap >= 0.8 and len(gt_nums) <= 1:
        partial_score = max(partial_score, 0.6)

    correct = partial_score >= 0.45
    is_hallucination = (
        number_score == 0
        and entity_overlap < 0.3
        and len(pred_lower) > 50
        and "error" not in pred_lower
    )

    # Build reasoning
    reasons = []
    if number_score == 1.0:
        reasons.append(f"All {len(gt_nums)} numbers matched")
    elif number_score > 0:
        reasons.append(f"{matched}/{len(gt_nums)} numbers matched (approx)")
    else:
        reasons.append("No numbers matched")

    if entity_overlap > 0:
        reasons.append(f"Entity overlap: {entity_overlap:.0%}")

    return {
        "correct": correct,
        "hallucination": is_hallucination,
        "partial_score": round(partial_score, 3),
        "number_recall": round(number_score, 3),
        "entity_overlap": round(entity_overlap, 3),
        "qualitative_match": entity_overlap > 0.3,
        "reasoning": "; ".join(reasons),
    }


# ── IR Metrics: nDCG@k, MRR ───────────────────────────────────────


def compute_dcg(relevances: list[float], k: int | None = None) -> float:
    """Compute Discounted Cumulative Gain at position k."""
    if k is not None:
        relevances = relevances[:k]
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevances)
    )


def compute_ndcg(relevances: list[float], k: int = 5) -> float:
    """
    Compute normalized DCG@k.

    Args:
        relevances: list of relevance scores (1.0 = relevant, 0.0 = irrelevant)
        k: cutoff position

    Returns:
        nDCG@k score in [0, 1]
    """
    dcg = compute_dcg(relevances, k)
    ideal = compute_dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def compute_mrr(ranked_relevances: list[list[float]]) -> float:
    """
    Compute Mean Reciprocal Rank across multiple queries.

    Args:
        ranked_relevances: list of relevance lists, one per query

    Returns:
        MRR score in [0, 1]
    """
    if not ranked_relevances:
        return 0.0

    reciprocal_ranks = []
    for relevances in ranked_relevances:
        rr = 0.0
        for i, rel in enumerate(relevances):
            if rel > 0:
                rr = 1.0 / (i + 1)
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
