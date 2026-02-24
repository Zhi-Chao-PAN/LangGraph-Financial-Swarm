import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State shared across all agents in the financial swarm."""
    messages: Annotated[List[BaseMessage], operator.add]
    next: Optional[str]
    sender: Optional[str]
