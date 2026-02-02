from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from src.tools.rag_tool import query_financial_rag
from src.core.types import AgentState
from src.core.prompts import Prompts
import re
import json
import uuid
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
        
        # Regex to find tool call
        # Pattern: TOOL_CALL: <name>\nARGS: <json>
        pattern = r"TOOL_CALL:\s*(query_financial_rag)\s*ARGS:\s*(\{.*?\})"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            try:
                args = robust_json_parse(args_str)
                # Manually construct AIMessage with tool_calls
                # ID is needed for ToolNode to match
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                tool_call = {
                    "name": tool_name.strip(),
                    "args": args,
                    "id": call_id,
                    "type": "tool_call" # LangChain specific? No, standard dict for tool_call
                }
                
                # We return an AIMessage that has tool_calls
                response.tool_calls = [tool_call]
                
            except Exception as e:
                log_agent_action("Researcher", "Error", f"Failed to parse tool args: {e}")
                # Return a system message to guide the model back
                return {
                    "messages": [response, SystemMessage(content=f"Error: Invalid JSON format in args. Please use valid JSON. Details: {e}")],
                    "sender": "Researcher"
                }
        
        return {
            "messages": [response],
            "sender": "Researcher"
        }
    
    return researcher_node
