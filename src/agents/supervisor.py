import re
from typing import List, Dict, Any, Callable
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
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

Respond with ONLY one word: Researcher, Quant, or FINISH"""


def create_supervisor_node(llm, members: List[str]) -> Callable:
    """
    Creates a Supervisor node with LLM-based routing + deterministic fallback.
    """

    def supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])

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
            return {"messages": [], "next": next_agent}

        if error_count >= 2:
            next_agent = "FINISH"
            log_agent_action("Supervisor", "Route", f"Decision: {next_agent} (repeated errors)")
            return {"messages": [], "next": next_agent}

        # Extract user question
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            elif hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break

        # Try LLM routing
        next_agent = _llm_route(llm, user_question, members)
        if next_agent is None:
            next_agent = _keyword_route(user_question, members)

        log_agent_action("Supervisor", "Route", f"Decision: {next_agent}")
        return {"messages": [], "next": next_agent}

    return supervisor_node


def _llm_route(llm, question: str, members: List[str]) -> str | None:
    """Try LLM-based routing. Returns None on failure."""
    if not question:
        return None
    try:
        response = llm.invoke([
            SystemMessage(content=ROUTING_PROMPT),
            HumanMessage(content=question),
        ])
        decision = _parse_routing(response.content, members)
        return decision
    except Exception as e:
        log_agent_action("Supervisor", "LLM Route Failed", str(e)[:100])
        return None


def _keyword_route(question: str, members: List[str]) -> str:
    """Deterministic keyword-based fallback routing."""
    q = question.lower()
    viz_keywords = ["chart", "plot", "graph", "visualiz", "draw", "diagram", "bar chart", "line chart"]
    if any(kw in q for kw in viz_keywords):
        return "Quant"
    return "Researcher"


def _parse_routing(content: str, members: List[str]) -> str:
    """
    Extract routing decision from LLM output.
    Looks for 'Next: <Role>' pattern or standalone agent name.
    Falls back to Researcher on ambiguity.
    """
    valid = set(members + ["FINISH"])

    # Try "Next: <agent>" pattern
    match = re.search(r"Next:\s*(\w+)", content, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for v in valid:
            if candidate.lower() == v.lower():
                return v

    # Try standalone word match
    clean = re.sub(r"<[^>]*>", "", content).strip()
    for v in valid:
        if v.lower() == clean.lower():
            return v

    # Try last word
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
