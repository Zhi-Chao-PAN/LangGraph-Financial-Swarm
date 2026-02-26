"""Tests for error recovery and fault tolerance."""

from unittest.mock import MagicMock, patch

from src.agents.researcher import create_researcher_node


class TestErrorRecovery:
    """Test graceful handling of various failure scenarios."""

    def test_rag_query_exception(self):
        """Researcher should handle RAG adapter exceptions gracefully."""
        mock_llm = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.query.side_effect = Exception("Connection timeout")

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "What was NVIDIA revenue?")]})

        assert result["sender"] == "Researcher"
        # Should still return a result, not crash
        assert "messages" in result

    def test_rag_partial_failure(self):
        """Multi-query should handle partial failures."""
        mock_llm = MagicMock()
        mock_adapter = MagicMock()

        call_count = 0

        def flaky_query(q):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Timeout")
            return {"model_answer": "Revenue was $130B", "source_nodes": ["src"]}

        mock_adapter.query.side_effect = flaky_query

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "Compare NVIDIA and Apple revenue")]})

        assert result["sender"] == "Researcher"

    def test_empty_rag_response(self):
        """Handle empty RAG responses."""
        mock_llm = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {"model_answer": "", "source_nodes": []}

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "What was NVIDIA revenue?")]})

        assert result["sender"] == "Researcher"

    def test_missing_source_nodes(self):
        """Handle response with missing source_nodes key."""
        mock_llm = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {"model_answer": "Revenue was $130B"}

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "What was NVIDIA revenue?")]})

        assert result["sender"] == "Researcher"


class TestCLICommands:
    """Test CLI command parsing and handling."""

    def test_handle_help(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/help", "")
        assert result is True

    def test_handle_quit(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/quit", "")
        assert result is False

    def test_handle_exit(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/exit", "")
        assert result is False

    def test_handle_unknown(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/unknown", "")
        assert result is True

    def test_handle_stats(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/stats", "")
        assert result is True

    def test_handle_history_empty(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/history", "")
        assert result is True

    def test_handle_templates(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/templates", "")
        assert result is True

    def test_handle_template_not_found(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/template", "nonexistent_template")
        assert result is True

    def test_handle_ask_empty(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/ask", "")
        assert result is True

    def test_handle_batch_no_file(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/batch", "")
        assert result is True

    def test_handle_batch_missing_file(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/batch", "nonexistent_file.txt")
        assert result is True

    def test_handle_fetch_no_ticker(self):
        from src.ui.cli import _handle_command

        result = _handle_command("/fetch", "")
        assert result is True
