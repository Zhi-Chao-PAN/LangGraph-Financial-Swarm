"""
Researcher agent with deterministic multi-query decomposition,
retrieval verification, and multi-turn conversation support.
"""

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.conversation import Conversation, is_followup, resolve_followup
from src.core.types import AgentState
from src.utils.robustness import log_agent_action

# Known companies in our dataset
KNOWN_COMPANIES = ["nvidia", "apple", "microsoft"]

REWRITE_PROMPT = """You are a financial query optimizer. Rewrite the user's question into 1-3 precise search queries for a financial RAG system.

Rules:
1. Decompose complex questions into simpler sub-queries
2. Include company name and fiscal year in each query
3. Use exact financial terms (revenue, net income, gross margin, etc.)
4. For comparison questions, create one query per entity
5. Output ONLY the queries, one per line, no numbering or bullets

Example:
User: Compare NVIDIA and Apple's gross margin in FY2025
Queries:
NVIDIA gross margin FY2025
Apple gross margin FY2025"""


def _rewrite_query(llm, question: str) -> list[str]:
    """Use LLM to decompose complex questions into sub-queries."""
    try:
        response = llm.invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(content=question),
            ]
        )
        content = response.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        queries = [line.strip() for line in content.split("\n") if line.strip() and not line.strip().startswith(("Queries:", "Query:"))]
        if queries:
            return queries[:3]
    except Exception as e:
        log_agent_action("Researcher", "Rewrite Failed", str(e)[:100])
    return [question]


def _detect_companies(question: str) -> list[str]:
    """Detect which companies are mentioned in the question."""
    q_lower = question.lower()
    return [c for c in KNOWN_COMPANIES if c in q_lower]


def _is_multi_entity_question(question: str) -> bool:
    """Determine if question requires data from multiple entities."""
    q_lower = question.lower()
    companies = _detect_companies(question)

    # Explicit multi-entity indicators
    multi_indicators = ["compare", "vs", "versus", "which company", "highest", "lowest", "between", "all three", "each company", "respectively"]
    has_indicator = any(ind in q_lower for ind in multi_indicators)

    return has_indicator or len(companies) >= 2


def _build_sub_queries(question: str) -> list[str]:
    """
    Deterministic sub-query generation for multi-entity questions.
    No LLM call — just template-based decomposition.
    """
    companies = _detect_companies(question)
    if len(companies) < 2:
        return [question]

    # Extract the metric being asked about by removing company names only (whole words)
    core = question
    for c in companies:
        # Remove company name as whole word (case-insensitive)
        core = re.sub(rf"\b{re.escape(c)}\b", "", core, flags=re.IGNORECASE)
        # Also remove possessive forms
        core = re.sub(rf"\b{re.escape(c)}'s\b", "", core, flags=re.IGNORECASE)

    # Remove comparison words but preserve financial terms
    for word in ["compare", "vs", "versus", "between", "which", "had the", "highest", "lowest", "or"]:
        core = re.sub(rf"\b{re.escape(word)}\b", "", core, flags=re.IGNORECASE)

    core = " ".join(core.split()).strip().strip("?.,:")

    # Generate one query per company with the core metric
    sub_queries = []
    for c in companies:
        if core:
            sub_queries.append(f"{c.upper()} {core}")

    # Always keep original question as fallback
    sub_queries.append(question)

    return sub_queries


def _verify_coverage(question: str, answer: str) -> list[str]:
    """
    Check if the answer covers all entities mentioned in the question.
    Returns list of missing entities.
    """
    companies = _detect_companies(question)
    missing = []
    answer_lower = answer.lower()
    for c in companies:
        if c not in answer_lower:
            missing.append(c)
    return missing


# Fiscal year pattern: FY2021, FY2025, FY 2023, etc.
_FY_PATTERN = re.compile(r"FY\s?(\d{4})", re.IGNORECASE)

# Year range patterns: FY2021 to FY2025, FY2021-FY2025, from 2021 to 2025
_RANGE_PATTERNS = [
    re.compile(r"FY\s?(\d{4})\s*(?:to|through|[-–])\s*FY\s?(\d{4})", re.IGNORECASE),
    re.compile(r"from\s+(?:FY\s?)?(\d{4})\s+to\s+(?:FY\s?)?(\d{4})", re.IGNORECASE),
]

# Trend indicators
_TREND_INDICATORS = [
    "trend",
    "growth",
    "cagr",
    "change over",
    "over time",
    "year over year",
    "yoy",
    "trajectory",
    "evolution",
    "increased",
    "decreased",
    "grew",
    "declined",
]


def _detect_fiscal_years(question: str) -> list[str]:
    """Extract all fiscal years mentioned in a question."""
    return _FY_PATTERN.findall(question)


def _detect_year_range(question: str) -> tuple[int, int] | None:
    """Detect a fiscal year range (e.g., FY2021 to FY2025). Returns (start, end) or None."""
    for pattern in _RANGE_PATTERNS:
        match = pattern.search(question)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _is_trend_question(question: str) -> bool:
    """Detect if a question asks about trends or changes over time."""
    q_lower = question.lower()
    return any(ind in q_lower for ind in _TREND_INDICATORS)


def _build_trend_queries(question: str) -> list[str]:
    """
    Build sub-queries for trend/time-range questions.
    Generates one query per year in the range for each company.
    """
    year_range = _detect_year_range(question)
    companies = _detect_companies(question)

    if not year_range:
        return [question]

    start, end = year_range
    if end - start > 10 or end - start < 1:
        return [question]

    # Extract the metric
    core = question
    for pattern in _RANGE_PATTERNS:
        core = pattern.sub("", core)
    for c in companies:
        core = re.sub(rf"\b{re.escape(c)}\b", "", core, flags=re.IGNORECASE)
        core = re.sub(rf"\b{re.escape(c)}'s\b", "", core, flags=re.IGNORECASE)
    for word in ["trend", "growth", "cagr", "over time", "trajectory", "what was", "calculate"]:
        core = re.sub(rf"\b{re.escape(word)}\b", "", core, flags=re.IGNORECASE)
    core = " ".join(core.split()).strip().strip("?.,:")

    sub_queries = []
    target_companies = companies if companies else [""]

    for company in target_companies:
        for year in range(start, end + 1):
            prefix = f"{company.upper()} " if company else ""
            sub_queries.append(f"{prefix}{core} FY{year}".strip())

    # Keep original as fallback
    sub_queries.append(question)

    log_agent_action("Researcher", "Trend Queries", f"{len(sub_queries)} queries for {start}-{end}")
    return sub_queries


def create_researcher_node(llm, conversation: Conversation | None = None) -> Callable[[AgentState], dict[str, Any]]:
    """
    Creates the Researcher node with deterministic multi-query,
    retrieval verification, and optional conversation context.
    """

    def researcher_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])

        # Extract the original user question
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break

        if not user_question:
            return {
                "messages": [AIMessage(content="No question found to research.")],
                "sender": "Researcher",
            }

        # Input validation
        if len(user_question) > 1000:
            user_question = user_question[:1000]
            log_agent_action("Researcher", "Truncate", "Query truncated to 1000 chars")

        # Resolve follow-up questions using conversation context
        if conversation and is_followup(user_question, conversation):
            original = user_question
            user_question = resolve_followup(user_question, conversation)
            log_agent_action("Researcher", "Follow-up", f"Resolved: '{original}' → enriched with context")

        from src.rag_adapter import adapter

        is_multi = _is_multi_entity_question(user_question)
        is_trend = _is_trend_question(user_question)

        if is_trend and _detect_year_range(user_question):
            sub_queries = _build_trend_queries(user_question)
            log_agent_action("Researcher", "Trend Query", f"{len(sub_queries)} queries (time-range)")
        elif is_multi:
            sub_queries = _build_sub_queries(user_question)
            log_agent_action("Researcher", "Multi-Query", f"{len(sub_queries)} queries (deterministic)")
        else:
            sub_queries = [user_question]

        # Execute all sub-queries
        all_answers = []
        all_sources = []

        NO_ANSWER_PATTERNS = [
            "does not include",
            "not found",
            "no information",
            "cannot find",
            "not available",
            "not provided",
        ]

        for sq in sub_queries:
            log_agent_action("Researcher", "RAG Query", f"Q: {sq}")
            try:
                result = adapter.query(sq)
                answer = result.get("model_answer", "No answer found.")
                sources = result.get("source_nodes", [])

                # Retry with simplified query if no answer
                if any(p in answer.lower() for p in NO_ANSWER_PATTERNS):
                    simplified = re.sub(r"(what was|how much|calculate|compare)\s+", "", sq, flags=re.IGNORECASE).strip()
                    if simplified != sq:
                        log_agent_action("Researcher", "Retry", f"Simplified: {simplified}")
                        result2 = adapter.query(simplified)
                        answer2 = result2.get("model_answer", "")
                        if not any(p in answer2.lower() for p in NO_ANSWER_PATTERNS):
                            answer = answer2
                            sources = result2.get("source_nodes", [])

                all_answers.append(answer)
                all_sources.extend(sources)
            except Exception as e:
                log_agent_action("Researcher", "Error", f"RAG query failed: {e}", level="error")
                all_answers.append(f"Error querying: {sq}")

        # Synthesize if multiple sub-queries
        if len(sub_queries) > 1 and len(all_answers) > 1:
            combined_context = "\n\n".join(f"[Sub-query: {sq}]\n{ans}" for sq, ans in zip(sub_queries, all_answers, strict=False))
            try:
                synthesis_prompt = (
                    f"Based on the following retrieved data, answer the original question.\n\n"
                    f"Original question: {user_question}\n\n"
                    f"Retrieved data:\n{combined_context}\n\n"
                    f"IMPORTANT: Include ALL specific numbers from ALL sub-queries in your answer. "
                    f"Do not omit any entity or data point. If the question asks about multiple "
                    f"companies or metrics, list each one with its value."
                )
                response = llm.invoke([SystemMessage(content=synthesis_prompt)])
                final_answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
            except Exception as e:
                final_answer = "\n\n".join(all_answers)
                log_agent_action("Researcher", "Synthesis Failed", str(e)[:100], level="warning")
        else:
            final_answer = all_answers[0] if all_answers else "No answer found."

        # Retrieval verification: check if all entities are covered
        missing = []
        if is_multi:
            missing = _verify_coverage(user_question, final_answer)
            if missing:
                log_agent_action("Researcher", "Coverage Gap", f"Missing: {missing}")
                # Targeted retrieval for missing entities
                for entity in missing:
                    fill_query = f"{entity.upper()} {user_question}"
                    log_agent_action("Researcher", "Fill Gap", f"Q: {fill_query}")
                    try:
                        result = adapter.query(fill_query)
                        fill_answer = result.get("model_answer", "")
                        if fill_answer and not any(p in fill_answer.lower() for p in NO_ANSWER_PATTERNS):
                            final_answer += f"\n\nAdditional data for {entity.upper()}: {fill_answer}"
                    except Exception:
                        pass

        # Compute confidence score
        confidence = _compute_confidence(
            final_answer,
            all_sources,
            sub_queries,
            all_answers,
            is_multi,
            missing if is_multi else [],
        )
        log_agent_action("Researcher", "Confidence", f"{confidence:.2f}")
        log_agent_action("Researcher", "Answer", final_answer[:200])

        answer_with_meta = f"{final_answer}\n\n[Confidence: {confidence:.2f}]"

        return {
            "messages": [AIMessage(content=answer_with_meta)],
            "sender": "Researcher",
        }

    return researcher_node


def _compute_confidence(
    answer: str,
    sources: list,
    sub_queries: list[str],
    sub_answers: list[str],
    is_multi: bool,
    missing_entities: list[str],
) -> float:
    """
    Compute confidence score (0.0-1.0) based on:
    - Source count (more sources = higher confidence)
    - Answer quality (no "not found" patterns)
    - Entity coverage (all entities answered)
    - Number presence (financial answers should have numbers)
    """
    score = 0.5  # base

    NO_ANSWER_PATTERNS = [
        "does not include",
        "not found",
        "no information",
        "cannot find",
        "not available",
        "not provided",
    ]

    # Source count bonus (up to +0.2)
    src_count = len(sources)
    score += min(src_count / 10, 0.2)

    # Answer quality (+0.15 if no "not found" patterns)
    if not any(p in answer.lower() for p in NO_ANSWER_PATTERNS):
        score += 0.15

    # Entity coverage (+0.1 if all covered)
    if is_multi and not missing_entities or not is_multi:
        score += 0.1

    # Number presence (+0.05 if answer contains numbers)
    if re.search(r"\d+[,.]?\d*", answer):
        score += 0.05

    return min(score, 1.0)
