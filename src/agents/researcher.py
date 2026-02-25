import re
from typing import Dict, Any, Callable, List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
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


def _rewrite_query(llm, question: str) -> List[str]:
    """Use LLM to decompose complex questions into sub-queries."""
    try:
        response = llm.invoke([
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=question),
        ])
        content = response.content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        queries = [line.strip() for line in content.split("\n")
                   if line.strip() and not line.strip().startswith(("Queries:", "Query:"))]
        if queries:
            return queries[:3]
    except Exception as e:
        log_agent_action("Researcher", "Rewrite Failed", str(e)[:100])
    return [question]


def _detect_companies(question: str) -> List[str]:
    """Detect which companies are mentioned in the question."""
    q_lower = question.lower()
    found = [c for c in KNOWN_COMPANIES if c in q_lower]
    return found


def _is_multi_entity_question(question: str) -> bool:
    """Determine if question requires data from multiple entities."""
    q_lower = question.lower()
    companies = _detect_companies(question)

    # Explicit multi-entity indicators
    multi_indicators = ["compare", "vs", "versus", "which company",
                        "highest", "lowest", "between", "all three",
                        "each company", "respectively"]
    has_indicator = any(ind in q_lower for ind in multi_indicators)

    return has_indicator or len(companies) >= 2


def _build_sub_queries(question: str) -> List[str]:
    """
    Deterministic sub-query generation for multi-entity questions.
    No LLM call — just template-based decomposition.
    """
    companies = _detect_companies(question)
    if len(companies) < 2:
        return [question]

    # Extract the metric being asked about
    q_lower = question.lower()

    # Remove company names and comparison words to get the core metric
    core = q_lower
    for c in companies:
        core = core.replace(c, "").replace(c.capitalize(), "")
    for word in ["compare", "vs", "versus", "between", "which", "had", "the",
                 "highest", "lowest", "and", "'s", "'s", "in", "company"]:
        core = core.replace(word, " ")
    core = " ".join(core.split()).strip().strip("?.,")

    # Generate one query per company
    sub_queries = [f"{c.upper()} {core}" for c in companies]

    # Also keep original question as fallback
    sub_queries.append(question)

    return sub_queries


def _verify_coverage(question: str, answer: str) -> List[str]:
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


def create_researcher_node(llm) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Creates the Researcher node with deterministic multi-query
    and retrieval verification.
    """

    def researcher_node(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])

        # Extract the original user question
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            elif hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break

        if not user_question:
            return {
                "messages": [AIMessage(content="No question found to research.")],
                "sender": "Researcher",
            }

        from src.rag_adapter import adapter

        is_multi = _is_multi_entity_question(user_question)

        if is_multi:
            sub_queries = _build_sub_queries(user_question)
            log_agent_action("Researcher", "Multi-Query",
                             f"{len(sub_queries)} queries (deterministic)")
        else:
            sub_queries = [user_question]

        # Execute all sub-queries
        all_answers = []
        all_sources = []

        NO_ANSWER_PATTERNS = [
            "does not include", "not found", "no information",
            "cannot find", "not available", "not provided",
        ]

        for sq in sub_queries:
            log_agent_action("Researcher", "RAG Query", f"Q: {sq}")
            try:
                result = adapter.query(sq)
                answer = result.get("model_answer", "No answer found.")
                sources = result.get("source_nodes", [])

                # Retry with simplified query if no answer
                if any(p in answer.lower() for p in NO_ANSWER_PATTERNS):
                    simplified = re.sub(r"(what was|how much|calculate|compare)\s+", "",
                                        sq, flags=re.IGNORECASE).strip()
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
                log_agent_action("Researcher", "Error", f"RAG query failed: {e}")
                all_answers.append(f"Error querying: {sq}")

        # Synthesize if multiple sub-queries
        if len(sub_queries) > 1 and len(all_answers) > 1:
            combined_context = "\n\n".join(
                f"[Sub-query: {sq}]\n{ans}" for sq, ans in zip(sub_queries, all_answers)
            )
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
                log_agent_action("Researcher", "Synthesis Failed", str(e)[:100])
        else:
            final_answer = all_answers[0] if all_answers else "No answer found."

        # Retrieval verification: check if all entities are covered
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

        log_agent_action("Researcher", "Answer", final_answer[:200])

        return {
            "messages": [AIMessage(content=final_answer)],
            "sender": "Researcher",
        }

    return researcher_node
