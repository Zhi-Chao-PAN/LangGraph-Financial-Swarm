import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import pytest
import os
from langchain_ollama import ChatOllama
from src.core.config import settings
from src.tools.plot_tool import create_plot
from main import main
from unittest.mock import MagicMock, patch

# Test Configuration
def test_config_loading():
    """Verify config loads correctly."""
    assert settings.LLM_MODEL is not None
    assert os.path.exists(settings.DATA_DIR)

# Test Tools
def test_plot_tool():
    """Verify plot tool logic (without verifying visual output)."""
    data = '[{"Year": 2023, "Revenue": 100}, {"Year": 2024, "Revenue": 150}]'
    # Mock log_agent_action to avoid file writes during test
    with patch("src.tools.plot_tool.log_agent_action"):
        result = create_plot.invoke({
            "data_str": data,
            "plot_type": "bar",
            "title": "Test Plot",
            "xlabel": "Year",
            "ylabel": "Revenue"
        })
        print(f"DEBUG TOOL OUTPUT: {result}")
        assert "Chart generated successfully" in result, f"Tool failed with: {result}"
        assert os.path.exists(os.path.join(settings.OUTPUT_DIR, "test_plot.png"))

# Test LLM Connection (Skip if Ollama not running)
@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping local LLM test in CI")
def test_llm_connection():
    """Verify we can talk to Ollama."""
    try:
        llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.LLM_BASE_URL)
        resp = llm.invoke("Hello")
        assert resp.content is not None
    except Exception as e:
        pytest.fail(f"Could not connect to Ollama: {e}")
