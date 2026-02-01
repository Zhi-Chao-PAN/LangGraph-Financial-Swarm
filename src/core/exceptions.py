class SwarmError(Exception):
    """Base exception for the Financial Swarm."""
    pass

class RAGConnectionError(SwarmError):
    """Raised when the RAG engine fails to connect or query."""
    pass

class ToolExecutionError(SwarmError):
    """Raised when a tool fails to execute correctly."""
    pass

class LLMGenerationError(SwarmError):
    """Raised when the LLM produces invalid or malformed output."""
    pass

class ConfigurationError(SwarmError):
    """Raised when the configuration is invalid."""
    pass
