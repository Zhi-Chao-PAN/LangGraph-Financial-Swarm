from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration management using Pydantic."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        extra="ignore",
    )

    # LLM Settings
    LLM_MODEL: str = Field(default="deepseek-r1:8b", description="Ollama Model Name")
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=1.0)
    LLM_BASE_URL: str = Field(default="http://localhost:11434")
    LLM_TIMEOUT: int = Field(default=300, gt=0)

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    # Paths
    BASE_DIR: Path = Path.cwd()
    DATA_DIR: Path = Field(default_factory=lambda: Path.cwd() / "data")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path.cwd() / "output")
    LOG_FILE: str = "agent_trace.log"

    # RAG Settings
    RAG_DATA_PATH: Path = Field(default_factory=lambda: Path.cwd() / "data" / "parsed" / "llamaparse" / "parsed.md")
    RAG_TOP_K: int = 5

    # Optional Keys
    LLAMA_CLOUD_API_KEY: str | None = None

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.LLM_TIMEOUT < 60:
            warnings.append(f"LLM_TIMEOUT={self.LLM_TIMEOUT}s is low; complex queries may timeout")

        if self.RAG_TOP_K < 3:
            warnings.append(f"RAG_TOP_K={self.RAG_TOP_K} is low; may miss relevant documents")

        if self.RAG_TOP_K > 20:
            warnings.append(f"RAG_TOP_K={self.RAG_TOP_K} is high; may slow down retrieval")

        if self.LLM_TEMPERATURE > 0.5:
            warnings.append(f"LLM_TEMPERATURE={self.LLM_TEMPERATURE} is high for factual financial queries")

        rag_path = Path(self.RAG_DATA_PATH)
        if not rag_path.exists():
            warnings.append(f"RAG data not found at {rag_path}; run ingestion first")

        return warnings


# Singleton instance
settings = Settings()
settings.ensure_dirs()
