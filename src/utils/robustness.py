import asyncio
import functools
import json
import logging
import time
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from typing import Any

from src.core.config import settings

logger = logging.getLogger("SwarmTracer")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s - [Robustness] - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1) -> Callable:
    """Decorator to retry a sync function with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        logger.error(f"{func.__name__} failed after {retries} retries: {e}")
                        raise
                    sleep = backoff_in_seconds * 2**attempt
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{retries}). Retrying in {sleep}s...")
                    time.sleep(sleep)
            return None

        return wrapper

    return decorator


def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1) -> Callable:
    """Decorator to retry an async function with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        logger.error(f"{func.__name__} failed after {retries} retries: {e}")
                        raise
                    sleep = backoff_in_seconds * 2**attempt
                    logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{retries}). Retrying in {sleep}s...")
                    await asyncio.sleep(sleep)
            return None

        return wrapper

    return decorator


def log_agent_action(agent_name: str, action: str, content: Any, level: str = "info") -> None:
    """
    Log significant agent actions for audit trails.

    Args:
        agent_name: Name of the agent (Supervisor, Researcher, Quant, etc.)
        action: Action type (Route, RAG Query, Answer, Error, etc.)
        content: Content preview
        level: Log level (debug, info, warning, error)
    """
    entry = {
        "agent": agent_name,
        "action": action,
        "content_preview": str(content)[:200] + "..." if len(str(content)) > 200 else str(content),
    }
    msg = json.dumps(entry)
    log_fn = getattr(logger, level, logger.info)
    log_fn(msg)
