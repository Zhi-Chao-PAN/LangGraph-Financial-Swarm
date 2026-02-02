# src/types.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, Union, List
from langchain_core.messages import BaseMessage
import operator

class AgentMetadata(TypedDict, total=False):
    """Metadata for execution tracking."""
    token_usage: int
    latency_ms: float
    step: int
    tool_calls: int

class AgentState(TypedDict):
    """
    TypedDict representing the state of the agent workflow.
    """
    # messages: Append-only list of chat messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # next: The next node to route to
    next: str
    
    # sender: The agent who sent the last message
    sender: str
    
    # metadata: Execution statistics and tracing info
    metadata: AgentMetadata

class FinancialData(TypedDict):
    """Structured representation of financial data for plotting."""
    label: str
    value: float
    year: int

# Type alias for tool outputs
ToolOutput = Union[str, Dict[str, Any]]
