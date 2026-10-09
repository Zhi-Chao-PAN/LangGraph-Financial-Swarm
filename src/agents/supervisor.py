"""
Supervisor agent with structured JSON routing and deterministic fallback.

Routes questions to Researcher or Quant agents, enforces max_rounds
to prevent infinite loops, and detects completion conditions.

SOTA enhancement:
- Structured JSON output format for reliable routing
- Eliminates fragile free-text parsing
"""

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.utils.robustness import log_agent_action

ROUTING_PROMPT = """You are a financial analysis supervisor. Route the user's question to the right specialist.

Available agents:
- Researcher: Retrieves financial data, answers factual questions, lookups, comparisons
- Quant: Creates charts, visualizations, plots, graphs
- FINISH: The question has been fully answered

Rules:
1. If the user asks for a chart, plot, graph, or visualization -> Quant
2. If the user asks a factual/analytical financial question -> Researcher
3. If an agent already provided a good answer -> FINISH
4. Default to Researcher if unsure

Respond with ONLY valid JSON in this exact format:
{"next": "<agent_name>", "reason": "<one-line reasoning>"}"""


def create_supervisor_node(llm, members: list[str], max_rounds: int = 10) -> Callable:
    """
    Creates a Supervisor node with LLM-based routing + deterministic fallback.
    Enforces max_rounds to prevent infinite loops.
    """
    round_counter = {"count": 0}

    def supervisor_node(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])

        # Safety: enforce max rounds
        round_counter["count"] += 1
        if round_counter["count"] > max_rounds:
            log_agent_action("Supervisor", "Route", f"Decision: FINISH (max_rounds={max_rounds} reached)")
            round_counter["count"] = 0
            return {"messages": [], "next": "FINISH"}

        # Check if any agent already responded with substantive content
        has_agent_response = False
        error_count = 0
        agent_response_count = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                content = msg.content
                if "Error" in content or "timed out" in content:
                    error_count += 1
                elif len(content) > 5:
                    has_agent_response = True
                    agent_response_count += 1

        if has_agent_response:
            next_agent = "FINISH"
            log_agent_action("Supervisor", "Route", f"Decision: {next_agent} (agent responded)")
            round_counter["count"] = 0
            return {"messages": [], "next": next_agent}

        if error_count >= 2:
            next_agent = "FINISH"
            log_agent_action("Supervisor", "Route", f"Decision: {next_agent} (repeated errors)")
            round_counter["count"] = 0
            return {"messages": [], "next": next_agent}

        # Extract user question
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break

        # Try LLM routing
        next_agent = _llm_route(llm, user_question, members)
        if next_agent is None:
            next_agent = _keyword_route(user_question, members)

        log_agent_action("Supervisor", "Route", f"Decision: {next_agent}")
        return {"messages": [], "next": next_agent}

    return supervisor_node


def _llm_route(llm, question: str, members: list[str]) -> str | None:
    """Try LLM-based routing. Returns None on failure."""
    if not question:
        return None
    try:
        response = llm.invoke(
            [
                SystemMessage(content=ROUTING_PROMPT),
                HumanMessage(content=question),
            ]
        )
        return _parse_routing(response.content, members)
    except Exception as e:
        log_agent_action("Supervisor", "LLM Route Failed", str(e)[:100])
        return None


def _keyword_route(question: str, members: list[str]) -> str:
    """Deterministic keyword-based fallback routing."""
    q = question.lower()
    viz_keywords = ["chart", "plot", "graph", "visualiz", "draw", "diagram", "bar chart", "line chart"]
    if any(kw in q for kw in viz_keywords):
        return "Quant"
    return "Researcher"


def _parse_routing(content: str, members: list[str]) -> str:
    """
    Extract routing decision from LLM output.
    Priority: JSON parse -> "Next: <Role>" pattern -> standalone word -> fallback.
    """
    valid = set(members + ["FINISH"])

    # Priority 1: Try JSON parse (structured output)
    try:
        # Strip markdown code blocks if present
        cleaned = re.sub(r"```json\s*", "", content)
        cleaned = re.sub(r"```\s*$", "", cleaned).strip()
        # Also strip <think> blocks
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

        data = json.loads(cleaned)
        if isinstance(data, dict):
            candidate = data.get("next", "").strip()
            for v in valid:
                if candidate.lower() == v.lower():
                    reason = data.get("reason", "")
                    if reason:
                        log_agent_action("Supervisor", "JSON Route", f"{v}: {reason[:100]}")
                    return v
    except (json.JSONDecodeError, AttributeError):
        pass

    # Priority 2: Try "Next: <agent>" pattern
    match = re.search(r"Next:\s*(\w+)", content, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for v in valid:
            if candidate.lower() == v.lower():
                return v

    # Priority 3: Try standalone word match
    clean = re.sub(r"<[^>]*>", "", content).strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    for v in valid:
        if v.lower() == clean.lower():
            return v

    # Priority 4: Try last word
    words = clean.split()
    if words:
        last_word = words[-1].strip(".,;:!?\"'")
        for v in valid:
            if last_word.lower() == v.lower():
                return v

    # Fallback
    if members:
        return members[0]
    return "FINISH"
