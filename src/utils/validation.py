import pandas as pd
import functools
import re
import json
import io
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def parse_dataframe(data_str: str) -> pd.DataFrame:
    """Parse supported JSON rows or CSV without treating JSON as CSV headers."""
    try:
        if data_str.lstrip().startswith(("[", "{")):
            data = json.loads(data_str)
            if not isinstance(data, list) or not all(
                isinstance(row, dict) for row in data
            ):
                raise ValueError("JSON data must be a list of dictionaries.")
            frame = pd.DataFrame(data)
        else:
            frame = pd.read_csv(io.StringIO(data_str))
        if frame.empty or len(frame.columns) < 2:
            raise ValueError("Plot data must contain rows and at least two columns.")
        return frame
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Data validation failed: {exc}") from exc


def validate_dataframe(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to validate that the input JSON string creates a valid DataFrame.
    Performs a 'dry run' parse to ensure data integrity before tool execution.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        data_str = kwargs.get("data_str")
        if not data_str:
            # Try positional arg
            if len(args) > 0:
                data_str = args[0]

        if not isinstance(data_str, str) or not data_str:
            raise ValueError("No string data_str provided to Tool.")

        parse_dataframe(data_str)

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
    clean = re.sub(r"[^\x20-\x7E]", "", clean)

    return clean
