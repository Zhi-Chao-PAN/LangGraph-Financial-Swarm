import time
import functools
import logging
import json
import asyncio
import inspect
from logging.handlers import RotatingFileHandler
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, cast
from src.core.config import settings

# Configure structured logger with rotation
logger = logging.getLogger("SwarmTracer")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    settings.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
)
handler.setFormatter(
    logging.Formatter("%(asctime)s - [Robustness] - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


P = ParamSpec("P")
R = TypeVar("R")


def retry_with_backoff(
    retries: int = 3, backoff_in_seconds: float = 1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to retry a function with exponential backoff.
    Useful for local LLM inference consistency.
    """

    if retries < 0 or backoff_in_seconds < 0:
        raise ValueError("Retries and backoff must be non-negative")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[P, Awaitable[object]], func)

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> object:
                for attempt in range(retries + 1):
                    try:
                        return await async_func(*args, **kwargs)
                    except Exception:
                        if attempt == retries:
                            raise
                        await asyncio.sleep(backoff_in_seconds * 2**attempt)
                raise AssertionError("Unreachable retry state")

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(
                            f"Function {func.__name__} failed after {retries} retries. Error: {e}"
                        )
                        raise
                    sleep = backoff_in_seconds * 2**x
                    logger.warning(
                        f"Function {func.__name__} failed (Attempt {x+1}/{retries}). Retrying in {sleep}s..."
                    )
                    time.sleep(sleep)
                    x += 1

        return wrapper

    return decorator


def log_agent_action(agent_name: str, action: str, content: Any) -> None:
    """
    Log significant agent actions for audit trails.
    """
    entry = {
        "agent": agent_name,
        "action": action,
        "content_preview": (
            str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
        ),
    }
    logger.info(json.dumps(entry))
