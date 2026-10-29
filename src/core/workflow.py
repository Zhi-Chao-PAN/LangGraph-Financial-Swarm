"""The shared graph used by the real CLI, offline demo, and integration tests."""

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.agents.chart_gen import create_quant_node
from src.agents.researcher import create_researcher_node
from src.agents.supervisor import create_supervisor_node
from src.core.types import AgentState

MEMBERS = ("Researcher", "Quant")


def route_tool_output(state: AgentState) -> str:
    sender = state.get("sender")
    if sender not in MEMBERS:
        raise ValueError(f"Unknown tool caller: {sender}")
    return sender


def route_member_output(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        raise ValueError("An agent must emit a message before routing")
    return "tools" if getattr(messages[-1], "tool_calls", None) else "Supervisor"


def build_swarm_graph(
    llm: BaseChatModel, tools: Sequence[BaseTool] | None = None
) -> CompiledStateGraph:
    if tools is None:
        from src.tools.plot_tool import create_plot
        from src.tools.rag_tool import query_financial_rag

        tools = [query_financial_rag, create_plot]

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", create_supervisor_node(llm, list(MEMBERS)))
    workflow.add_node("Researcher", create_researcher_node(llm))
    workflow.add_node("Quant", create_quant_node(llm))
    workflow.add_node("tools", ToolNode(list(tools)))
    workflow.add_conditional_edges(
        "Supervisor",
        lambda state: state["next"],
        {"Researcher": "Researcher", "Quant": "Quant", "FINISH": END},
    )
    workflow.add_conditional_edges(
        "tools", route_tool_output, {member: member for member in MEMBERS}
    )
    for member in MEMBERS:
        workflow.add_conditional_edges(
            member, route_member_output, {"tools": "tools", "Supervisor": "Supervisor"}
        )
    workflow.add_edge(START, "Supervisor")
    return workflow.compile()
