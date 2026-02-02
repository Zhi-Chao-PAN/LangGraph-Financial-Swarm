import argparse
import sys
import asyncio
import logging
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.core.types import AgentState
from src.agents.supervisor import create_supervisor_node
from src.agents.researcher import create_researcher_node
from src.agents.chart_gen import create_quant_node
from src.core.config import settings
from src.utils.observability import Observability
from src.utils.validation import sanitize_input

# Disable standard logging in favor of Rich
logging.basicConfig(level=logging.CRITICAL)

async def main():
    """
    Main asynchronous entry point for the Financial Swarm.
    Initializes agents and the LangGraph workflow.
    """
    # Parse CLI Arguments
    parser = argparse.ArgumentParser(description="LangGraph Financial Swarm")
    parser.add_argument("--query", type=str, default="Compare NVIDIA's revenue growth from 2023 to 2024.", help="The financial question to answer.")
    args = parser.parse_args()
    
    # Sanitize Input (Security Best Practice)
    clean_query = sanitize_input(args.query)
    
    Observability.start_trace()
    
    # Initialize LLM (Configured in src/core/config.py)
    llm = ChatOllama(
        model=settings.LLM_MODEL, 
        temperature=settings.LLM_TEMPERATURE, 
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT
    )

    # 1. Create Nodes
    from langgraph.prebuilt import ToolNode, tools_condition
    from src.tools.rag_tool import query_financial_rag
    from src.tools.plot_tool import create_plot
    
    tools = [query_financial_rag, create_plot]
    tool_node = ToolNode(tools)
    
    members: list[str] = ["Researcher", "Quant"]
    supervisor_node = create_supervisor_node(llm, members)
    researcher_node = create_researcher_node(llm)
    quant_node = create_quant_node(llm)
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Quant", quant_node)
    workflow.add_node("tools", tool_node)

    # 2. Define Edges
    # Workflow: 
    # Supervisor -> Researcher/Quant -> tools -> Researcher/Quant -> Supervisor
    
    # Conditional edge from Supervisor to members
    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"]
    )
    
    # Conditional edge from members to tools or back to Supervisor
    for member in members:
        workflow.add_conditional_edges(member, tools_condition)
        
    # Edge from tools back to the agent that called them
    # Note: In a swarm, usually tools return to the sender. 
    # tools_condition will route AIMessages with tool_calls to 'tools'
    # and AIMessages without them to '__end__' or another node.
    # Here, 'tools' node should return to the sender. 
    # We can use a custom router or multiple tool nodes.
    # Simplified: tools always go back to the caller. 
    # But ToolNode doesn't know the sender. LangGraph 0.2+ tools_condition handles this?
    # Actually, we should use a custom logic to route back to the sender.
    def route_tool_output(state: AgentState):
        return state["sender"]

    workflow.add_edge("tools", route_tool_output)

    # Edge from members (when NO tool call) to Supervisor
    # This is handled by tools_condition if we route the 'else' (END) to Supervisor.
    # Let's override tools_condition behavior or implement simple version:
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "Supervisor"

    for member in members:
         workflow.add_conditional_edges(
            member,
            should_continue,
            {
                "tools": "tools",
                "Supervisor": "Supervisor"
            }
        )

    workflow.add_edge(START, "Supervisor")

    graph = workflow.compile()
    
    print(f"Goal: {args.query}")
    steps = 0

    from src.utils.observability import console
    
    # 3. Run Graph Asynchronously
    with console.status("[bold blue]Agents are collaborating...[/]", spinner="dots"):
        async for s in graph.astream(
            {"messages": [("user", clean_query)]},
            {"recursion_limit": 20}
        ):
            steps += 1
            if "__end__" not in s:
                for key, val in s.items():
                    if "messages" in val:
                        msg = val['messages'][-1]
                        sender = val.get("sender", "System")
                        # Render via Observability
                        Observability.trace_agent(sender, msg.content, metadata={"step": steps})
    
    Observability.final_report(steps)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
