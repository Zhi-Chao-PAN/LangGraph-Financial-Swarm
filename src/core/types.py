import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all agents in the financial swarm."""

    messages: Annotated[list[BaseMessage], operator.add]
    next: str | None
    sender: str | None
