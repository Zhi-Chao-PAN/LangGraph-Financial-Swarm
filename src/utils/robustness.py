import time
import asyncio
import functools
import logging
import json
from logging.handlers import RotatingFileHandler
from typing import Any, Callable
from src.core.config import settings

logger = logging.getLogger("SwarmTracer")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s - [Robustness] - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """Decorator to retry a sync function with exponential backoff."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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

        return wrapper

    return decorator


def async_retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """Decorator to retry an async function with exponential backoff."""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
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

        return wrapper

    return decorator


def log_agent_action(agent_name: str, action: str, content: Any):
    """Log significant agent actions for audit trails."""
    entry = {
        "agent": agent_name,
        "action": action,
        "content_preview": str(content)[:200] + "..." if len(str(content)) > 200 else str(content),
    }
    logger.info(json.dumps(entry))
