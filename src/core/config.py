from pydantic_settings import BaseSettings
from pydantic import ConfigDict, Field
from pathlib import Path


class Settings(BaseSettings):
    """Typed configuration management using Pydantic."""

    model_config = ConfigDict(
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
    RAG_DATA_PATH: Path = Field(
        default_factory=lambda: Path.cwd() / "data" / "parsed" / "llamaparse" / "parsed.md"
    )
    RAG_TOP_K: int = 5

    # Optional Keys
    LLAMA_CLOUD_API_KEY: str | None = None

    def ensure_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_dirs()
