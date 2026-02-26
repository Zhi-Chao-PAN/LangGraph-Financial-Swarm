"""
Multi-LLM backend support.
Supports Ollama (local), OpenAI API, DeepSeek API, and custom endpoints.
Auto-detection and graceful fallback between backends.
"""

from dataclasses import dataclass
from enum import Enum

import requests

from src.utils.robustness import log_agent_action


class LLMBackend(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


@dataclass
class LLMConfig:
    """Configuration for an LLM backend."""

    backend: LLMBackend
    model: str
    base_url: str
    api_key: str | None = None
    temperature: float = 0.1
    timeout: int = 120
    max_tokens: int = 4096

    @property
    def display_name(self) -> str:
        return f"{self.backend.value}:{self.model}"


# Preset configurations
PRESETS: dict[str, LLMConfig] = {
    "ollama-deepseek": LLMConfig(
        backend=LLMBackend.OLLAMA,
        model="deepseek-r1:8b",
        base_url="http://localhost:11434",
    ),
    "ollama-qwen": LLMConfig(
        backend=LLMBackend.OLLAMA,
        model="qwen2.5:7b",
        base_url="http://localhost:11434",
    ),
    "openai-gpt4o": LLMConfig(
        backend=LLMBackend.OPENAI,
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        api_key="",  # User must set
    ),
    "deepseek-chat": LLMConfig(
        backend=LLMBackend.DEEPSEEK,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="",  # User must set
    ),
}


def check_backend_available(config: LLMConfig) -> bool:
    """Check if an LLM backend is available."""
    if config.backend == LLMBackend.OLLAMA:
        try:
            resp = requests.get(f"{config.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                # Check if model is available (with or without tag)
                model_base = config.model.split(":")[0]
                return any(model_base in m for m in models)
        except Exception:
            return False
    elif config.backend in (LLMBackend.OPENAI, LLMBackend.DEEPSEEK):
        return bool(config.api_key)
    elif config.backend == LLMBackend.CUSTOM:
        try:
            resp = requests.get(config.base_url, timeout=5)
            return resp.status_code < 500
        except Exception:
            return False
    return False


def select_best_backend(
    preferred: str | None = None,
    configs: dict[str, LLMConfig] | None = None,
) -> LLMConfig | None:
    """
    Select the best available LLM backend.
    Tries preferred first, then falls back through available options.
    """
    if configs is None:
        configs = PRESETS

    # Try preferred first
    if preferred and preferred in configs:
        cfg = configs[preferred]
        if check_backend_available(cfg):
            log_agent_action("LLMSelector", "Selected", f"{cfg.display_name} (preferred)")
            return cfg

    # Fallback order: local first, then cloud
    fallback_order = [
        LLMBackend.OLLAMA,
        LLMBackend.DEEPSEEK,
        LLMBackend.OPENAI,
        LLMBackend.CUSTOM,
    ]

    for backend_type in fallback_order:
        for _name, cfg in configs.items():
            if cfg.backend == backend_type and check_backend_available(cfg):
                log_agent_action("LLMSelector", "Fallback", f"{cfg.display_name}")
                return cfg

    log_agent_action("LLMSelector", "Error", "No LLM backend available", level="error")
    return None


def list_backends(configs: dict[str, LLMConfig] | None = None) -> list[dict]:
    """List all configured backends with availability status."""
    if configs is None:
        configs = PRESETS

    result = []
    for name, cfg in configs.items():
        available = check_backend_available(cfg)
        result.append(
            {
                "name": name,
                "backend": cfg.backend.value,
                "model": cfg.model,
                "base_url": cfg.base_url,
                "available": available,
                "has_api_key": bool(cfg.api_key),
            }
        )
    return result
