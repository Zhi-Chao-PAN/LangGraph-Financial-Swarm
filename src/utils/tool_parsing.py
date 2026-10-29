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
    def parse_tool_call(
        content: str, tools_whitelist: List[str], sender: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parses the content for a specific tool call pattern.

        Args:
            content (str): The raw text output from the LLM.
            tools_whitelist (List[str]): List of valid tool names to look for.
            sender (str): Name of the agent calling the tool (for logging).

        Returns:
            Optional[Dict[str, Any]]: A tool_call dictionary format compatible with LangChain/LangGraph, or None.
        """
        match = re.search(
            r"TOOL_CALL:\s*([A-Za-z_]\w*)\s*ARGS:\s*", content, re.IGNORECASE
        )
        if not match:
            return None
        allowed = {name.casefold(): name for name in tools_whitelist}
        tool_name = allowed.get(match.group(1).casefold())
        if tool_name is None:
            return None
        tail = content[match.end() :].lstrip()
        if not tail.startswith("{"):
            return None

        # A non-greedy regex truncates nested objects and braces inside JSON strings.
        depth = 0
        quote: str | None = None
        escaped = False
        end = None
        for index, char in enumerate(tail):
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            return None
        try:
            args = robust_json_parse(tail[:end])
            if not isinstance(args, dict):
                return None
            return {
                "name": tool_name,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }
        except ValueError as exc:
            log_agent_action(
                sender, "Error", f"Failed to parse tool args for {tool_name}: {exc}"
            )
            return None
