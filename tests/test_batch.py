"""Tests for batch query executor."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.core.batch import BatchReport, BatchResult, load_queries_from_file, run_batch


class TestBatchResult:
    def test_basic(self):
        r = BatchResult(question="Test?", answer="Yes", latency_s=1.0)
        assert r.question == "Test?"
        assert r.error is None

    def test_with_error(self):
        r = BatchResult(question="Fail?", answer="", latency_s=5.0, error="Timeout")
        assert r.error == "Timeout"


class TestBatchReport:
    def test_empty(self):
        report = BatchReport()
        assert report.total == 0
        assert report.avg_latency == 0.0
        assert report.avg_confidence == 0.0

    def test_with_results(self):
        report = BatchReport(
            results=[
                BatchResult("Q1", "A1", 1.0, confidence=0.8),
                BatchResult("Q2", "A2", 2.0, confidence=0.9),
                BatchResult("Q3", "", 5.0, error="Timeout"),
            ],
            total_time_s=8.0,
        )
        assert report.total == 3
        assert report.success_count == 2
        assert report.error_count == 1
        assert abs(report.avg_latency - 1.5) < 0.01
        assert abs(report.avg_confidence - 0.85) < 0.01

    def test_summary(self):
        report = BatchReport(
            results=[BatchResult("Q1", "A1", 1.0, confidence=0.8)],
            total_time_s=1.0,
        )
        s = report.summary()
        assert s["total"] == 1
        assert s["success"] == 1
        assert s["errors"] == 0

    def test_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report = BatchReport(
                results=[BatchResult("Q1", "A1", 1.0)],
                total_time_s=1.0,
            )
            report.save(path)
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["summary"]["total"] == 1
            assert len(data["results"]) == 1


class TestRunBatch:
    def test_basic_batch(self):
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "model_answer": "Revenue was $130,497M [Confidence: 0.85]",
            "source_nodes": ["src1", "src2"],
        }
        report = run_batch(["Q1", "Q2", "Q3"], adapter=mock_adapter)
        assert report.total == 3
        assert report.success_count == 3
        assert report.error_count == 0
        assert mock_adapter.query.call_count == 3

    def test_batch_with_errors(self):
        mock_adapter = MagicMock()
        mock_adapter.query.side_effect = [
            {"model_answer": "Answer", "source_nodes": []},
            Exception("Timeout"),
            {"model_answer": "Answer2", "source_nodes": []},
        ]
        report = run_batch(["Q1", "Q2", "Q3"], adapter=mock_adapter)
        assert report.success_count == 2
        assert report.error_count == 1

    def test_progress_callback(self):
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {"model_answer": "A", "source_nodes": []}
        progress = []
        run_batch(
            ["Q1", "Q2"],
            adapter=mock_adapter,
            progress_callback=lambda i, t, q: progress.append((i, t, q)),
        )
        assert len(progress) == 2
        assert progress[0] == (0, 2, "Q1")

    def test_confidence_extraction(self):
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {
            "model_answer": "Revenue $130B [Confidence: 0.92]",
            "source_nodes": [],
        }
        report = run_batch(["Q1"], adapter=mock_adapter)
        assert report.results[0].confidence == 0.92


class TestLoadQueries:
    def test_load_text_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.txt"
            path.write_text("What is revenue?\nWhat is margin?\n# comment\n", encoding="utf-8")
            queries = load_queries_from_file(path)
            assert len(queries) == 2
            assert queries[0] == "What is revenue?"

    def test_load_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.json"
            path.write_text('["Q1", "Q2", "Q3"]', encoding="utf-8")
            queries = load_queries_from_file(path)
            assert len(queries) == 3

    def test_load_nonexistent(self):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_queries_from_file("/nonexistent/queries.txt")

    def test_load_empty_lines_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queries.txt"
            path.write_text("Q1\n\n\nQ2\n\n", encoding="utf-8")
            queries = load_queries_from_file(path)
            assert len(queries) == 2
