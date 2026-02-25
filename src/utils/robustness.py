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


def check_ollama_health(base_url: str | None = None, timeout: int = 5) -> dict[str, Any]:
    """
    Check if Ollama service is reachable and has models loaded.

    Returns:
        {"healthy": bool, "models": list[str], "error": str | None}
    """
    import requests

    url = base_url or settings.LLM_BASE_URL

    try:
        # Check service is up
        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return {"healthy": False, "models": [], "error": f"HTTP {resp.status_code}"}

        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        log_agent_action("Health", "Ollama", f"OK, {len(models)} models: {models}")
        return {"healthy": True, "models": models, "error": None}
    except requests.ConnectionError:
        msg = f"Ollama not reachable at {url}"
        log_agent_action("Health", "Ollama", msg, level="error")
        return {"healthy": False, "models": [], "error": msg}
    except Exception as e:
        msg = f"Health check failed: {e}"
        log_agent_action("Health", "Ollama", msg, level="error")
        return {"healthy": False, "models": [], "error": msg}


def graceful_llm_call(llm, messages: list, fallback: str = "LLM unavailable. Please try again later.") -> str:
    """
    Call LLM with graceful degradation. Returns fallback message on failure.
    """
    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        log_agent_action("Graceful", "LLM Fallback", str(e)[:200], level="warning")
        return fallback
