from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.rag_tool import query_financial_rag
from src.core.types import AgentState
from src.core.prompts import Prompts
from typing import Dict, Any, Callable, Sequence
from langchain_core.messages import BaseMessage
from src.utils.parsing import robust_json_parse
from src.utils.robustness import log_agent_action
from langchain_core.messages import SystemMessage

def create_researcher_node(llm: ChatOllama) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Creates the Researcher node for financial data retrieval.
    
    Args:
        llm (ChatOllama): The local LLM instance.
        
    Returns:
        Callable[[AgentState], Dict[str, Any]]: The state-graph compatible node function.
    """
    
    def researcher_node(state: AgentState) -> Dict[str, Any]:
        """
        Process the state and generate a research response.
        
        Args:
            state (AgentState): The current graph state.
            
        Returns:
            Dict[str, Any]: Identify of the sender and updated messages.
        """
        messages = [HumanMessage(content=Prompts.RESEARCHER_SYSTEM)] + state["messages"]
        response = llm.invoke(messages)
        content = response.content
        
        from src.utils.tool_parsing import ToolParser
        
        # Use centralized ToolParser (DRY Principle)
        tool_call = ToolParser.parse_tool_call(content, ["query_financial_rag"], "Researcher")
        
        if tool_call:
            response.tool_calls = [tool_call]
        else:
            # If no tool call found, check if we need to handle parsing errors or just return the response
            # efficient-handling: ToolParser logs errors internally.
            pass
        
        return {
            "messages": [response],
            "sender": "Researcher"
        }
    
    return researcher_node
