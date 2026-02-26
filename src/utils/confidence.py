"""
Confidence scoring and source attribution for RAG answers.
Provides traceability: which sources contributed to each answer,
and how confident the system is in the response.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SourceAttribution:
    """A single source that contributed to an answer."""

    source_id: str
    content_snippet: str
    relevance_score: float
    company: str = ""
    section: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "snippet": self.content_snippet[:200],
            "relevance": round(self.relevance_score, 3),
            "company": self.company,
            "section": self.section,
        }


@dataclass
class ConfidenceResult:
    """Confidence assessment for an answer."""

    score: float  # 0.0 - 1.0
    level: str  # high / medium / low
    reasons: list[str] = field(default_factory=list)
    sources: list[SourceAttribution] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "level": self.level,
            "reasons": self.reasons,
            "sources": [s.to_dict() for s in self.sources],
        }

    def to_markdown(self) -> str:
        emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(self.level, "⚪")
        lines = [f"{emoji} Confidence: {self.score:.0%} ({self.level})"]
        if self.reasons:
            for r in self.reasons:
                lines.append(f"  - {r}")
        if self.sources:
            lines.append(f"  Sources: {len(self.sources)} documents")
        return "\n".join(lines)


def score_confidence(
    answer: str,
    query: str,
    source_nodes: list | None = None,
    retrieval_scores: list[float] | None = None,
) -> ConfidenceResult:
    """
    Score confidence of an answer based on multiple signals.

    Signals:
    1. Answer quality (length, specificity, hedging)
    2. Source relevance (retrieval scores)
    3. Numeric evidence (contains specific numbers)
    4. Query-answer alignment (key terms overlap)
    """
    score = 0.5  # Base score
    reasons: list[str] = []
    sources: list[SourceAttribution] = []

    # 1. Answer quality
    if not answer or len(answer.strip()) < 10:
        score -= 0.3
        reasons.append("Answer too short or empty")
    elif len(answer) > 100:
        score += 0.1
        reasons.append("Detailed answer")

    # Hedging detection
    hedging_patterns = [
        r"\bi('m| am) not sure\b",
        r"\bpossibly\b",
        r"\bmaybe\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bunclear\b",
        r"\bI don'?t know\b",
        r"\bnot available\b",
    ]
    hedging_count = sum(1 for p in hedging_patterns if re.search(p, answer, re.IGNORECASE))
    if hedging_count > 0:
        score -= 0.1 * min(hedging_count, 3)
        reasons.append(f"Hedging language detected ({hedging_count} patterns)")

    # 2. Numeric evidence
    numbers = re.findall(r"\$?[\d,]+\.?\d*[BMK%]?", answer)
    if len(numbers) >= 2:
        score += 0.15
        reasons.append(f"Contains {len(numbers)} numeric values")
    elif len(numbers) == 1:
        score += 0.05
        reasons.append("Contains numeric evidence")

    # 3. Source relevance
    if retrieval_scores:
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        if avg_score > 0.8:
            score += 0.15
            reasons.append(f"High source relevance ({avg_score:.2f})")
        elif avg_score > 0.5:
            score += 0.05
            reasons.append(f"Moderate source relevance ({avg_score:.2f})")
        else:
            score -= 0.1
            reasons.append(f"Low source relevance ({avg_score:.2f})")

    # 4. Query-answer alignment
    query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
    answer_terms = set(re.findall(r"\b\w{3,}\b", answer.lower()))
    if query_terms:
        overlap = len(query_terms & answer_terms) / len(query_terms)
        if overlap > 0.5:
            score += 0.1
            reasons.append(f"Good query-answer alignment ({overlap:.0%})")

    # 5. Build source attributions
    if source_nodes:
        for i, node in enumerate(source_nodes[:5]):
            content = str(node) if not isinstance(node, dict) else node.get("text", str(node))
            rel_score = retrieval_scores[i] if retrieval_scores and i < len(retrieval_scores) else 0.5

            # Try to extract company and section
            company = ""
            section = ""
            company_patterns = ["NVIDIA", "Apple", "Microsoft", "Alphabet", "Amazon"]
            for cp in company_patterns:
                if cp.lower() in content.lower():
                    company = cp
                    break

            section_match = re.search(r"(?:^|\n)#+\s*(.+)", content)
            if section_match:
                section = section_match.group(1).strip()[:50]

            sources.append(
                SourceAttribution(
                    source_id=f"src_{i + 1}",
                    content_snippet=content[:200],
                    relevance_score=rel_score,
                    company=company,
                    section=section,
                )
            )

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Determine level
    if score >= 0.7:
        level = "high"
    elif score >= 0.4:
        level = "medium"
    else:
        level = "low"

    return ConfidenceResult(
        score=score,
        level=level,
        reasons=reasons,
        sources=sources,
    )
