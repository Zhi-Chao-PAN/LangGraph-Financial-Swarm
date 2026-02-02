from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import re

# Import Types and Prompts
from src.core.types import AgentState
from src.core.prompts import Prompts

from typing import List, Sequence, Callable, Dict, Any
from src.core.constants import ROLE_FINISH

def create_supervisor_node(llm: ChatOllama, members: List[str]) -> Callable[[AgentState], Dict[str, Any]]:
    """
    Creates the Supervisor node function (Regex Augmented for Robustness).
    
    Args:
        llm (ChatOllama): The local LLM instance.
        members (List[str]): List of worker agent names.

    Returns:
        Callable[[AgentState], Dict[str, Any]]: The graph code for the supervisor.
    """
    options = [ROLE_FINISH] + members
    
    # 构建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", Prompts.SUPERVISOR_SYSTEM),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Based on the conversation, who should act next? "
            "Respond with 'Next: <Role>' where <Role> is one of {options}.",
        ),
    ]).partial(options=str(options))

    # Regex parser with robust fallback
    def parse_route(ai_message):
        text = ai_message.content
        # 0. Strip <think> blocks for reasoning models (DeepSeek-R1, etc.)
        text_to_parse = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        
        # Debug log for routing
        from src.utils.robustness import log_agent_action
        log_agent_action("Supervisor", "Thought", text_to_parse[:200].strip()) # Log truncated thought 
        
        # 1. Try strict regex on stripped content
        match = re.search(r"Next:\s*(Researcher|Quant|FINISH)", text_to_parse, re.IGNORECASE)
        if match:
             return {"next": match.group(1).title() if match.group(1).upper() != "FINISH" else "FINISH"}
        
        # 2. Heuristic fallback
        if "FINISH" in text.upper() and len(text) < 50:
             return {"next": "FINISH"}
        
        # 3. Default safety net: If inconclusive, ask Researcher for more info
        return {"next": "Researcher"}

    supervisor_chain = prompt | llm | parse_route

    return supervisor_chain
