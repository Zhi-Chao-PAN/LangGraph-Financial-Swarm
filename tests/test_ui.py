"""Tests for Rich CLI UI components."""

import json
import tempfile
from pathlib import Path

from src.ui import print_banner, print_status_dashboard
from src.ui.display import (
    export_results,
    format_answer_panel,
    print_answer,
    print_cache_stats,
    print_comparison_table,
    print_eval_results,
    print_query_stats,
)


class TestBanner:
    def test_banner_no_crash(self, capsys):
        print_banner()
        # Just verify it doesn't crash

    def test_status_dashboard_healthy(self, capsys):
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.LLM_MODEL = "deepseek-r1:8b"
        mock_settings.EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        mock_settings.RAG_DATA_PATH = Path("/nonexistent/path.md")
        mock_settings.DATA_DIR = Path(tempfile.gettempdir())

        health = {"healthy": True, "models": ["deepseek-r1:8b"], "error": None}
        print_status_dashboard(health, mock_settings)

    def test_status_dashboard_unhealthy(self, capsys):
        from unittest.mock import MagicMock

        mock_settings = MagicMock()
        mock_settings.LLM_MODEL = "deepseek-r1:8b"
        mock_settings.EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
        mock_settings.RAG_DATA_PATH = Path("/nonexistent/path.md")
        mock_settings.DATA_DIR = Path(tempfile.gettempdir())

        health = {"healthy": False, "models": [], "error": "Connection refused"}
        print_status_dashboard(health, mock_settings)


class TestAnswerPanel:
    def test_basic_panel(self):
        panel = format_answer_panel("What is revenue?", "Revenue was $130,497M", confidence=0.85, latency=2.3)
        assert panel is not None
        assert panel.title is not None

    def test_panel_with_citations(self):
        answer = "Revenue was $130,497M\n\n**Citations**:\n[Source 1] Revenue table..."
        panel = format_answer_panel("Revenue?", answer, confidence=0.9)
        assert panel is not None

    def test_panel_long_question_truncated(self):
        long_q = "A" * 200
        panel = format_answer_panel(long_q, "Answer")
        # Title should be truncated
        assert panel is not None

    def test_print_answer_no_crash(self, capsys):
        print_answer("Test question?", "Test answer.", confidence=0.75, latency=1.5)


class TestTables:
    def test_comparison_table(self, capsys):
        data = [
            {"Company": "NVIDIA", "Revenue": "$130,497M", "Growth": "114%"},
            {"Company": "Apple", "Revenue": "$391,035M", "Growth": "2%"},
        ]
        print_comparison_table(data, title="Revenue Comparison")

    def test_comparison_table_empty(self, capsys):
        print_comparison_table([], title="Empty")

    def test_eval_results(self, capsys):
        results = [
            {"question": "NVIDIA revenue?", "correct": True, "partial_score": 0.95, "hallucination": False},
            {"question": "Apple margin?", "correct": False, "partial_score": 0.3, "hallucination": True},
        ]
        print_eval_results(results)

    def test_query_stats(self, capsys):
        stats = {"total_queries": 10, "avg_latency": 2.5, "cache_hit_rate": 0.3, "avg_confidence": 0.85}
        print_query_stats(stats)


class TestExport:
    def test_export_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results = [{"question": "Test?", "answer": "Yes", "correct": True}]
            export_results(results, str(path), fmt="json")
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert len(data) == 1
            assert data[0]["question"] == "Test?"

    def test_export_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.md"
            results = [{"question": "Revenue?", "answer": "$130B", "correct": True, "partial_score": 0.9}]
            export_results(results, str(path), fmt="markdown")
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "Revenue?" in content
            assert "$130B" in content

    def test_export_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "results.json"
            export_results([{"q": "test"}], str(path))
            assert path.exists()


class TestCacheStats:
    def test_cache_stats_no_dir(self, capsys):
        print_cache_stats(Path("/nonexistent/cache"))

    def test_cache_stats_empty(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            print_cache_stats(Path(tmpdir))
