import pandas as pd
import functools
import re
from typing import Any, Callable

def validate_dataframe(func: Callable) -> Callable:
    """
    Decorator to validate that the input JSON string creates a valid DataFrame.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        data_str = kwargs.get('data_str')
        if not data_str:
            # Try positional arg
            if len(args) > 0:
                data_str = args[0]
        
        if not data_str:
             raise ValueError("No data_str provided to Tool.")

        # Basic Sanity Check
        if len(data_str) < 10:
             raise ValueError("Data string too short to be valid JSON/CSV.")
             
        # Content validation happens inside the tool, but this ensures non-empty input
        return func(*args, **kwargs)
    return wrapper

def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input to prevent injection or overly long queries.
    """
    # 1. Strip whitespace
    clean = user_input.strip()
    
    # 2. Length check (Max 500 chars for financial query)
    if len(clean) > 500:
        clean = clean[:500]
        
    # 3. Remove non-printable characters (basic filtering)
    clean = re.sub(r'[^\x20-\x7E]', '', clean)
    
    return clean
