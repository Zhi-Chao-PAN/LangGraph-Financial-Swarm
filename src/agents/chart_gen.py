"""
Chart generation agent (Quant Analyst).

Generates financial visualizations by extracting data from
researcher answers and producing bar/line/pie/stacked_bar charts.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from src.core.types import AgentState
from src.utils.robustness import log_agent_action

CHART_PROMPT = """You are a financial data visualization expert. Based on the context, generate chart parameters as JSON.

Output ONLY valid JSON with these keys:
{
  "data_str": "[{...}]",
  "plot_type": "bar|line|pie|stacked_bar",
  "title": "Chart Title",
  "xlabel": "X-axis label",
  "ylabel": "Y-axis label"
}

Rules:
- data_str must be a JSON array string with consistent keys
- For comparisons: use grouped bar chart
- For trends over time: use line chart
- For composition/share: use pie chart
- Include actual numbers from the context, not placeholders"""


def create_quant_node(llm) -> Callable[[AgentState], dict[str, Any]]:
    """
    Creates the Quant node for financial data visualization.
    Supports bar, line, pie, and stacked_bar charts.
    """

    def quant_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])

        # Gather context from recent messages
        context_parts = []
        for msg in messages[-5:]:
            if hasattr(msg, "content") and msg.content:
                context_parts.append(msg.content)
        context = "\n".join(context_parts)

        # Extract user question
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break

        prompt = f"{CHART_PROMPT}\n\nUser question: {user_question}\n\nContext:\n{context[:2000]}"

        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            content = response.content

            # Remove <think> blocks
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                from src.tools.plot_tool import create_plot

                params = json.loads(json_match.group())

                log_agent_action("Quant", "Chart", f"Type: {params.get('plot_type', 'bar')}, Title: {params.get('title', 'N/A')}")

                result = create_plot.invoke(params)
                final = f"Visualization complete. {result}"
            else:
                final = f"Could not generate chart parameters. Raw output: {content[:200]}"
                log_agent_action("Quant", "Warning", "No JSON found in LLM output")
        except json.JSONDecodeError as e:
            final = f"Invalid JSON from LLM: {e}"
            log_agent_action("Quant", "Error", final)
        except Exception as e:
            final = f"Chart generation failed: {e}"
            log_agent_action("Quant", "Error", final)

        return {
            "messages": [AIMessage(content=final)],
            "sender": "Quant",
        }

    return quant_node
