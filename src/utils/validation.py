
import pandas as pd
import functools
import re
import json
import io
from typing import Any, Callable

def validate_dataframe(func: Callable) -> Callable:
    """
    Decorator to validate that the input JSON string creates a valid DataFrame.
    Performs a 'dry run' parse to ensure data integrity before tool execution.
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

        # 1. Basic Length Check
        if len(data_str) < 10:
             raise ValueError("Data string too short to be valid JSON/CSV.")
             
        # 2. Schema/Format Validation (Dry Run)
        # We assume the data is JSON list of dicts, as per prompt instructions
        try:
            # Use standard json first for speed, or robust logic if needed. 
            # Since this is a validation step for a tool that expects clean JSON from the code generator,
            # we can use the robust parser if available, or just standard json.
            # Let's try standard json to be strict, or pd.read_json
            
            # Attempt to parse into a DataFrame to verify structure
            # Handle user passing a string representation of a list
            try:
                data = json.loads(data_str)
            except Exception:
                # If json fails, maybe it's CSV? Unlikely for this specific tool intent but good for robustness
                raise ValueError("Invalid JSON format.")
                
            if not isinstance(data, list):
                 raise ValueError("Data must be a list of dictionaries.")
                 
            if len(data) == 0:
                 raise ValueError("Data list is empty.")
                 
            # Dry run dataframe creation
            df = pd.DataFrame(data)
            if df.empty:
                 raise ValueError("DataFrame is empty after parsing.")
                 
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}")

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
