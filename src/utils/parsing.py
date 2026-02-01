
import json
import re
from typing import Any, Dict, List, Union

def robust_json_parse(json_str: str) -> Union[Dict, List, Any]:
    """
    Robustly parse JSON strings, handling common LLM errors like:
    - Single quotes instead of double quotes
    - Trailing commas
    - Python literals (True/False/None)
    - Missing quotes around keys
    """
    if not json_str:
        raise ValueError("Empty JSON string")
        
    # 1. Try standard parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # 2. Clean up common issues
    # Replace single quotes with double quotes
    # Be careful not to replace apostrophes in text, this is a heuristic
    # A safer approach is to use a dedicated library like json5 if available, 
    # but here we implement basic heuristics to stay dependency-light or use regex.
    
    cleaned = json_str
    
    # Python constants to JSON
    cleaned = cleaned.replace("True", "true").replace("False", "false").replace("None", "null")
    
    # Try parsing again
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. Aggressive cleanup using regex to fix quotes (simple case)
    # This is risky but often needed for regex-extracted args like {'a': 1}
    try:
        # Replace single quotes wrapping keys/values with double quotes
        # This is a naive implementation; for production, use a library like `demjson` or `json5`
        # Here we assume the user might not have `json5` installed unless we add it to requirements.
        # We will try to replace ' with " if " is not present effectively.
        if "'" in cleaned and '"' not in cleaned:
             cleaned = cleaned.replace("'", '"')
             return json.loads(cleaned)
    except:
        pass
        
    # 4. Final attempt: json5 if installed (User suggestion)
    try:
        import json5
        return json5.loads(json_str)
    except ImportError:
        pass
        
    # 5. Fail
    raise ValueError(f"Failed to parse JSON: {json_str[:50]}...")
