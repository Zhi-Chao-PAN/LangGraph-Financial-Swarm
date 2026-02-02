
import json
import re
from typing import Any, Dict, List, Union

def robust_json_parse(json_str: str) -> Union[Dict, List, Any]:
    """
    Robustly parse JSON strings, prioritizing standard libraries over heuristics.
    
    Strategy:
    1. Try `json5` (supports trailing commas, comments, single quotes).
    2. Fallback to standard `json`.
    3. Last resort heuristics for common LLM malformations.
    """
    if not json_str:
        raise ValueError("Empty JSON string")

    # 1. Try json5 (Best for LLM outputs like single quotes/trailing commas)
    try:
        import json5
        return json5.loads(json_str)
    except (ImportError, Exception):
        # json5 not available or failed
        pass

    # 2. Standard JSON fallback
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 3. Last Resort Heuristics (Explicitly marked)
    # Only attempted if professional parsers fail. 
    # LLMs sometimes output Python dictionaries (e.g. {'key': True}).
    cleaned = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
    if "'" in cleaned and '"' not in cleaned:
        cleaned = cleaned.replace("'", '"')
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON content: {json_str[:50]}...")
