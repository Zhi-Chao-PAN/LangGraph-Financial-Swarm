import pytest
from unittest.mock import patch, MagicMock
from src.rag_adapter import RAGAdapter
from src.core.exceptions import RAGConnectionError
from src.core.config import settings

@pytest.mark.asyncio
async def test_rag_network_failure():
    """Verify that RAGAdapter fallback triggers on network error."""
    # We mock RAGAdapter init to avoid real diskcache/ollama connection during unit test
    with patch("src.rag_adapter.RAGAdapter._load_data") as mock_load:
        mock_load.return_value = MagicMock()
        
        adapter = RAGAdapter()
        
        # Mock the query engine to raise an exception
        adapter.query_engine = MagicMock()
        adapter.query_engine.aquery.side_effect = Exception("Connection Timeout")
        
        # This should handle the exception gracefully and return fallback
        result = await adapter.aquery("What is NVIDIA's revenue?")
        
        # Check for fallback signature
        assert "Fallback Data" in result["model_answer"]
        assert result["latency_s"] == 0.0

def test_config_load():
    """Verify Settings load correctly."""
    assert settings.LLM_MODEL is not None
    assert settings.LLM_TIMEOUT > 0
