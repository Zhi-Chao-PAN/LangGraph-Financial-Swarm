"""A scripted, synthetic run through the production graph, with real chart export."""

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.workflow import build_swarm_graph
from src.tools.plot_tool import create_plot

DEMO_QUERY = "Compare ExampleCo's synthetic revenue in 2023 and 2024 and plot it."
DEMO_ROWS = [{"Year": 2023, "Revenue": 100}, {"Year": 2024, "Revenue": 120}]


@tool
def demo_retrieval(question: str) -> str:
    """Retrieve the fixed synthetic ExampleCo fixture; no external service is used."""
    return json.dumps({"synthetic": True, "company": "ExampleCo", "rows": DEMO_ROWS})


# The graph sees the production tool name while the implementation remains offline.
demo_retrieval.name = "query_financial_rag"


async def run_demo(max_steps: int = 20) -> dict[str, Any]:
    plot_args = {
        "data_str": json.dumps(DEMO_ROWS),
        "plot_type": "bar",
        "title": "ExampleCo synthetic revenue",
        "xlabel": "Year",
        "ylabel": "Synthetic units",
    }
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="Next: Researcher"),
            AIMessage(
                content='TOOL_CALL: query_financial_rag\nARGS: {"question":"ExampleCo revenue"}'
            ),
            AIMessage(content="Synthetic revenue was 100 in 2023 and 120 in 2024."),
            AIMessage(content="Next: Quant"),
            AIMessage(content=f"TOOL_CALL: create_plot\nARGS: {json.dumps(plot_args)}"),
            AIMessage(
                content="ExampleCo's synthetic revenue increased by 20%. The chart uses the fixture values."
            ),
            AIMessage(content="Next: FINISH"),
        ]
    )
    graph = build_swarm_graph(llm, tools=[demo_retrieval, create_plot])
    steps: list[str] = []
    tool_results: list[dict[str, str]] = []
    answer = ""
    async for update in graph.astream(
        {"messages": [("user", DEMO_QUERY)]}, {"recursion_limit": max_steps}
    ):
        for node, output in update.items():
            steps.append(node)
            for message in output.get("messages", []):
                if isinstance(message, ToolMessage):
                    if message.status == "error" or str(message.content).startswith(
                        "Error"
                    ):
                        raise RuntimeError(
                            f"Offline demo tool failed: {message.content}"
                        )
                    tool_results.append(
                        {"name": message.name or "", "result": str(message.content)}
                    )
                elif isinstance(message, AIMessage) and not message.tool_calls:
                    answer = str(message.content)
    if [item["name"] for item in tool_results] != [
        "query_financial_rag",
        "create_plot",
    ]:
        raise RuntimeError("Offline demo did not complete both tool calls")
    return {
        "synthetic": True,
        "model_backend": "scripted",
        "query": DEMO_QUERY,
        "answer": answer,
        "steps": steps,
        "tools": tool_results,
        "benchmark_claim": "none",
    }
