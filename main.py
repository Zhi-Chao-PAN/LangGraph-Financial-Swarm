import argparse
import sys
import asyncio
import logging

from langgraph.graph import StateGraph, START, END

from src.core.types import AgentState
from src.agents.supervisor import create_supervisor_node
from src.agents.researcher import create_researcher_node
from src.agents.chart_gen import create_quant_node
from src.core.config import settings
from src.core.llm import DirectOllamaChat
from src.utils.observability import Observability, console
from src.utils.validation import sanitize_input

logging.basicConfig(level=logging.CRITICAL)


def build_graph(llm):
    """Build and compile the agent graph."""
    members = ["Researcher", "Quant"]
    supervisor_node = create_supervisor_node(llm, members)
    researcher_node = create_researcher_node(llm)
    quant_node = create_quant_node(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Quant", quant_node)

    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {"Researcher": "Researcher", "Quant": "Quant", "FINISH": END},
    )
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Quant", "Supervisor")
    workflow.add_edge(START, "Supervisor")

    return workflow.compile()


async def run_query(graph, query: str):
    """Run a single query through the graph."""
    steps = 0
    final_answer = ""

    with console.status("[bold blue]Agents are collaborating...[/]", spinner="dots"):
        async for s in graph.astream(
            {"messages": [("user", query)]},
            {"recursion_limit": 15},
        ):
            steps += 1
            if "__end__" not in s:
                for key, val in s.items():
                    if "messages" in val and val["messages"]:
                        msg = val["messages"][-1]
                        sender = val.get("sender", key)
                        final_answer = msg.content
                        Observability.trace_agent(
                            sender, msg.content, metadata={"step": steps}
                        )

    return final_answer, steps


async def interactive_mode(graph):
    """Interactive REPL mode."""
    console.print("[bold green]Interactive mode. Type 'quit' to exit.[/]\n")
    while True:
        try:
            query = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        query = sanitize_input(query)
        answer, steps = await run_query(graph, query)
        Observability.final_report(steps)
        console.print()


async def main():
    parser = argparse.ArgumentParser(description="LangGraph Financial Swarm")
    parser.add_argument("--query", type=str, default=None, help="Financial question to answer")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    args = parser.parse_args()

    Observability.start_trace()

    llm = DirectOllamaChat(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )

    if not llm.ping():
        console.print("[bold red]Error: Cannot connect to Ollama server.[/]")
        console.print(f"[dim]Expected at {settings.LLM_BASE_URL}. Run: ollama serve[/]")
        sys.exit(1)

    graph = build_graph(llm)

    if args.interactive:
        await interactive_mode(graph)
    else:
        query = args.query or "Compare NVIDIA's revenue growth from 2023 to 2024."
        clean_query = sanitize_input(query)
        console.print(f"[bold]Q: {clean_query}[/]\n")
        answer, steps = await run_query(graph, clean_query)
        Observability.final_report(steps)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
