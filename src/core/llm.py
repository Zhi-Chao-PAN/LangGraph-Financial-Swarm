"""
Drop-in replacement for ChatOllama that uses requests directly.
Workaround for ollama Python client 502 bug with older Ollama versions.
"""

import requests
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field


class DirectOllamaChat(BaseChatModel):
    """Chat model that calls Ollama REST API directly via requests."""

    model: str = "deepseek-r1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout: int = 300

    @property
    def _llm_type(self) -> str:
        return "direct-ollama"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, tuple) and len(msg) == 2:
                result.append({"role": msg[0], "content": msg[1]})
            else:
                result.append({"role": "user", "content": str(msg.content)})
        return result

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("message", {}).get("content", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def ping(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
