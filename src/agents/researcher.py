import re
from typing import Dict, Any, Callable, List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from src.core.types import AgentState
from src.utils.robustness import log_agent_action

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
        # Remove <think>...</think> blocks from deepseek
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Parse lines
        queries = [line.strip() for line in content.split("\n")
                   if line.strip() and not line.strip().startswith(("Queries:", "Query:"))]
        if queries:
            return queries[:3]  # Max 3 sub-queries
    except Exception as e:
        log_agent_action("Researcher", "Rewrite Failed", str(e)[:100])
    return [question]


def create_researcher_node(llm) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Creates the Researcher node with query rewriting for complex questions.
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

        # Determine if query rewriting is needed
        complex_indicators = ["compare", "vs", "versus", "which company",
                              "highest", "lowest", "between", "all three",
                              "each company", "respectively"]
        # "and" alone is too broad — only trigger if combined with company/entity context
        has_multi_entity = any(ind in user_question.lower() for ind in complex_indicators)
        if not has_multi_entity and " and " in user_question.lower():
            # Only rewrite if "and" connects entities (heuristic: multiple company names)
            company_names = ["nvidia", "apple", "microsoft"]
            mentioned = sum(1 for c in company_names if c in user_question.lower())
            has_multi_entity = mentioned >= 2

        if has_multi_entity:
            sub_queries = _rewrite_query(llm, user_question)
            log_agent_action("Researcher", "Query Rewrite",
                             f"Original: {user_question} -> {len(sub_queries)} sub-queries")
        else:
            sub_queries = [user_question]

        # Execute all sub-queries and merge results
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

                # If RAG couldn't find answer, retry with simplified query
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
                log_agent_action("Researcher", "Error", f"RAG query failed: {e}")
                all_answers.append(f"Error querying: {sq}")

        # If multiple sub-queries, synthesize a combined answer
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

        log_agent_action("Researcher", "Answer", final_answer[:200])

        return {
            "messages": [AIMessage(content=final_answer)],
            "sender": "Researcher",
        }

    return researcher_node
