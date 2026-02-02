import re
import uuid
import json
from typing import Dict, Any, Optional, List
from langchain_core.messages import AIMessage
from src.utils.parsing import robust_json_parse
from src.utils.robustness import log_agent_action

class ToolParser:
    """
    Standardized parser for extracting tool calls from LLM outputs.
    Adheres to the DRY principle by centralizing regex logic.
    """
    
    @staticmethod
    def parse_tool_call(content: str, tools_whitelist: List[str], sender: str) -> Optional[Dict[str, Any]]:
        """
        Parses the content for a specific tool call pattern.
        
        Args:
            content (str): The raw text output from the LLM.
            tools_whitelist (List[str]): List of valid tool names to look for.
            sender (str): Name of the agent calling the tool (for logging).
            
        Returns:
            Optional[Dict[str, Any]]: A tool_call dictionary format compatible with LangChain/LangGraph, or None.
        """
        # Pattern: TOOL_CALL: <name>\nARGS: <json>
        # Supports multi-line JSON and various spacing
        pattern = r"TOOL_CALL:\s*(" + "|".join(tools_whitelist) + r")\s*ARGS:\s*(\{.*?\})"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            try:
                args = robust_json_parse(args_str)
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                
                tool_call = {
                    "name": tool_name.strip(),
                    "args": args,
                    "id": call_id,
                    "type": "tool_call"
                }
                return tool_call
                
            except Exception as e:
                log_agent_action(sender, "Error", f"Failed to parse tool args for {tool_name}: {e}")
                pass
                
        return None
