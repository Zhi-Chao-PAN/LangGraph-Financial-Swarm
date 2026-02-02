from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.core.types import AgentState
from src.core.prompts import Prompts
from typing import Dict, Any, Callable, Sequence 
from src.utils.parsing import robust_json_parse
from src.utils.robustness import log_agent_action
from langchain_core.messages import SystemMessage

def create_quant_node(llm: ChatOllama) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Creates the Quant node for data visualization.
    
    Args:
        llm (ChatOllama): The local LLM instance.
        
    Returns:
        Callable[[AgentState], Dict[str, Any]]: The state-graph compatible node function.
    """
    
    def quant_node(state: AgentState) -> Dict[str, Any]:
        """
        Process the state and generate a visualization response.

        Args:
            state (AgentState): The current graph state.

        Returns:
            Dict[str, Any]: Updated state with Quant's response.
        """
        messages = [HumanMessage(content=Prompts.QUANT_SYSTEM)] + state["messages"]
        response = llm.invoke(messages)
        content = response.content
        
        from src.utils.tool_parsing import ToolParser
        
        # Use centralized ToolParser (DRY Principle)
        tool_call = ToolParser.parse_tool_call(content, ["create_plot"], "Quant")
        
        if tool_call:
            response.tool_calls = [tool_call]
        
        return {
            "messages": [response],
            "sender": "Quant"
        }
    
    return quant_node
