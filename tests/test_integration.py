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

# Test Invariant: Supervisor Output Validity
def test_supervisor_validity_under_noise():
    """
    Research Grade Test: Verify that the Supervisor acts as a deterministic firewall.
    Even if the LLM emits garbage (stochastic element), the routing policy (deterministic element)
    must ALWAYS return a valid state transition from the defined set.
    This proves the 'Routing Stability' claim in the README.
    """
    from src.agents.supervisor import create_supervisor_node
    
    # 1. Simulate a completely broken LLM response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "I am a stochastic parrot behaving badly with no structured output."
    
    # 2. Create supervisor with standard roles
    members = ["Researcher", "Quant"]
    supervisor_chain = create_supervisor_node(mock_llm, members)
    
    # 3. Invoke with empty state
    result = supervisor_chain.invoke({"messages": []})
    
    # 4. Assert Invariant: Output MUST be a valid role or fall back to a safe default
    # The README claims: "policy with asymmetric fallback... biasing towards information-seeking agents"
    valid_transitions = members + ["FINISH"]
    
    print(f"DEBUG: Broken Input -> Supervisor Output: {result}")
    
    assert "next" in result, "Supervisor output must contain routing key 'next'."
    assert result["next"] in valid_transitions, \
        f"Supervisor emitted invalid transition '{result['next']}'. Allowed: {valid_transitions}"
    
    # 5. Verify asymmetric bias (should fall back to Researcher for safety)
    assert result["next"] == "Researcher", "Ideally, ambiguous input should bias towards data retrieval (Researcher)."
