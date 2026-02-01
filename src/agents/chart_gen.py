from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.plot_tool import create_plot
from src.core.types import AgentState
from src.core.prompts import Prompts
import re
import json
import uuid

from typing import Dict, Any, Callable, Sequence 

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
        
        # Regex to find tool call
        pattern = r"TOOL_CALL:\s*(create_plot)\s*ARGS:\s*(\{.*?\})"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            try:
                args = json.loads(args_str)
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                tool_call = {
                    "name": tool_name.strip(),
                    "args": args,
                    "id": call_id,
                    "type": "tool_call"
                }
                response.tool_calls = [tool_call]
            except Exception as e:
                print(f"Failed to parse tool args: {e}")
        
        return {
            "messages": [response],
            "sender": "Quant"
        }
    
    return quant_node
