
import json
import re
from typing import Any, Dict, List, Union

def robust_json_parse(json_str: str) -> Union[Dict, List, Any]:
    """
    Robustly parse JSON strings, strictly prioritizing professional libraries.
    
    1. Try `json5` (Handles trailing commas, comments, single quotes).
    2. Fallback to standard `json`.
    3. Last resort: Heuristic repair for common LLM malformations (e.g. invalid boolean literals).
    """
    if not json_str:
        raise ValueError("Empty JSON string")

    # 1. First Priority: json5 (Professional standard for loose JSON)
    try:
        import json5
        return json5.loads(json_str)
    except (ImportError, Exception):
        pass

    # 2. Second Priority: Standard JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 3. Last Resort: Heuristic Fallback
    # CAUTION: String replacement is dangerous. Only done if standard parsing fails.
    try:
        # Resolve Python literals to JSON
        cleaned = json_str.replace("True", "true").replace("False", "false").replace("None", "null")
        # heuristic: replace single quotes if no double quotes exist
        if "'" in cleaned and '"' not in cleaned:
             cleaned = cleaned.replace("'", '"')
        return json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON content: {json_str[:50]}...")
