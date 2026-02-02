import time
import functools
import logging
import json
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict
from src.core.config import settings

# Configure structured logger with rotation
logger = logging.getLogger("SwarmTracer")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(settings.LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
handler.setFormatter(logging.Formatter('%(asctime)s - [Robustness] - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Decorator to retry a function with exponential backoff.
    Useful for local LLM inference consistency.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Function {func.__name__} failed after {retries} retries. Error: {e}")
                        raise
                    sleep = (backoff_in_seconds * 2 ** x)
                    logger.warning(f"Function {func.__name__} failed (Attempt {x+1}/{retries}). Retrying in {sleep}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

def log_agent_action(agent_name: str, action: str, content: Any):
    """
    Log significant agent actions for audit trails.
    """
    entry = {
        "agent": agent_name,
        "action": action,
        "content_preview": str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
    }
    logger.info(json.dumps(entry))
