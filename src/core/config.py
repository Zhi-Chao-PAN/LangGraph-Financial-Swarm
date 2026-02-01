from pydantic_settings import BaseSettings
from pathlib import Path
from pydantic import Field

class Settings(BaseSettings):
    """
    Strict, typed configuration management using Pydantic.
    """
    # LLM Settings
    LLM_MODEL: str = Field(default="deepseek-r1:8b", description="Ollama Model Name")
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=1.0)
    LLM_BASE_URL: str = Field(default="http://localhost:11434")
    LLM_TIMEOUT: int = Field(default=120, gt=0)

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    # Paths (Computed)
    BASE_DIR: Path = Path.cwd()
    DATA_DIR: Path = Field(default_factory=lambda: Path.cwd() / "data")
    OUTPUT_DIR: Path = Field(default_factory=lambda: Path.cwd() / "output")
    LOG_FILE: str = "agent_trace.log"
    
    # RAG Settings
    RAG_DATA_PATH: Path = Field(default_factory=lambda: Path.cwd() / "data" / "parsed" / "llamaparse" / "parsed.md")
    RAG_TOP_K: int = 3
    
    # Optional Keys
    LLAMA_CLOUD_API_KEY: str | None = None

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore" # Allow extra keys in .env

    def ensure_dirs(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # RAG_DATA_PATH directory is now managed by ingestion process, not auto-created empty.

# Singleton instance
settings = Settings()
settings.ensure_dirs()
