import functools
import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd


def validate_dataframe(func: Callable) -> Callable:
    """
    Decorator to validate that the input JSON string creates a valid DataFrame.
    Performs a dry-run parse before tool execution.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        data_str = kwargs.get("data_str")
        if not data_str and len(args) > 0:
            data_str = args[0]

        if not data_str:
            raise ValueError("No data_str provided to Tool.")

        if len(data_str) < 10:
            raise ValueError("Data string too short to be valid JSON/CSV.")

        try:
            data = json.loads(data_str)
        except Exception as err:
            raise ValueError("Invalid JSON format.") from err

        if not isinstance(data, list):
            raise ValueError("Data must be a list of dictionaries.")

        if len(data) == 0:
            raise ValueError("Data list is empty.")

        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("DataFrame is empty after parsing.")

        return func(*args, **kwargs)

    return wrapper


def sanitize_input(user_input: str) -> str:
    """
    Sanitize user input to prevent injection or overly long queries.
    Preserves Unicode characters (Chinese, Japanese, etc.).
    """
    clean = user_input.strip()

    if len(clean) > 500:
        clean = clean[:500]

    # Remove control characters but keep all printable Unicode
    return re.sub(r"[\x00-\x1f\x7f]", "", clean)

