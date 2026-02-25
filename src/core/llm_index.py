"""
Custom LlamaIndex LLM that calls Ollama REST API directly via requests.
Bypasses the ollama Python client which has compatibility issues.
"""

import requests
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback


class DirectOllamaLLM(CustomLLM):
    """LlamaIndex-compatible LLM using direct REST calls to Ollama."""

    model: str = "deepseek-r1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    request_timeout: int = 300
    context_window: int = 8192

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=2048,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.request_timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "")
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs):
        # Fallback to non-streaming
        return self.complete(prompt, **kwargs)
