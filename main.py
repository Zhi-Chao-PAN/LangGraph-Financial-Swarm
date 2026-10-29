import argparse
import sys
import asyncio
import logging

from src.core.config import settings
from src.utils.observability import Observability
from src.utils.validation import sanitize_input

# Disable standard logging in favor of Rich
logging.basicConfig(level=logging.CRITICAL)


async def main() -> None:
    """
    Main asynchronous entry point for the Financial Swarm.
    Initializes agents and the LangGraph workflow.
    """
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="LangGraph Financial Swarm")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="The financial question to answer.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the fixed synthetic offline graph and real chart export",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the offline demo result as JSON"
    )
    parser.add_argument(
        "--max-steps", type=int, default=20, help="Graph recursion limit (1-100)"
    )
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 100:
        parser.error("--max-steps must be between 1 and 100")
    if args.demo:
        if args.query is not None:
            parser.error("--demo uses a fixed synthetic query; omit --query")
        import json
        from src.core.demo import run_demo

        result = await run_demo(max_steps=args.max_steps)
        print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
        return
    if args.json:
        parser.error("--json is currently available with --demo")
    query = (
        args.query
        if args.query is not None
        else "Compare NVIDIA's revenue growth from 2023 to 2024."
    )

    # Sanitize Input (Security Best Practice)
    clean_query = sanitize_input(query)
    if not clean_query:
        parser.error("--query must contain printable text")

    Observability.start_trace()

    # Initialize LLM (Configured in src/core/config.py)
    from langchain_ollama import ChatOllama
    from src.core.workflow import build_swarm_graph

    llm = ChatOllama(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )

    graph = build_swarm_graph(llm)

    print(f"Goal: {query}")
    steps = 0

    from src.utils.observability import console

    # 3. Run Graph Asynchronously
    with console.status("[bold blue]Agents are collaborating...[/]", spinner="dots"):
        async for s in graph.astream(
            {"messages": [("user", clean_query)]}, {"recursion_limit": args.max_steps}
        ):
            steps += 1
            if "__end__" not in s:
                for key, val in s.items():
                    if "messages" in val:
                        msg = val["messages"][-1]
                        sender = val.get("sender", "System")
                        # Render via Observability
                        Observability.trace_agent(
                            sender, msg.content, metadata={"step": steps}
                        )

    Observability.final_report(steps)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
