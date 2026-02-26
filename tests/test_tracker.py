"""Tests for query performance tracker."""

import json
import tempfile
from pathlib import Path

from src.utils.tracker import PerformanceTracker, QueryRecord


class TestQueryRecord:
    def test_basic_record(self):
        rec = QueryRecord(question="Test?", latency_s=1.5, confidence=0.85, source_count=3)
        assert rec.question == "Test?"
        assert rec.latency_s == 1.5
        assert rec.confidence == 0.85
        assert rec.source_count == 3
        assert not rec.cache_hit
        assert rec.error is None
        assert rec.timestamp > 0

    def test_cache_hit(self):
        rec = QueryRecord(question="Cached?", latency_s=0.01, cache_hit=True)
        assert rec.cache_hit

    def test_error_record(self):
        rec = QueryRecord(question="Fail?", latency_s=5.0, error="Timeout")
        assert rec.error == "Timeout"


class TestPerformanceTracker:
    def test_empty_tracker(self):
        t = PerformanceTracker()
        assert t.total_queries == 0
        assert t.avg_latency == 0.0
        assert t.p95_latency == 0.0
        assert t.cache_hit_rate == 0.0
        assert t.avg_confidence == 0.0
        assert t.error_rate == 0.0

    def test_record_and_stats(self):
        t = PerformanceTracker()
        t.record("Q1", 1.0, confidence=0.8)
        t.record("Q2", 2.0, confidence=0.9)
        t.record("Q3", 3.0, confidence=0.7)
        assert t.total_queries == 3
        assert abs(t.avg_latency - 2.0) < 0.01
        assert abs(t.avg_confidence - 0.8) < 0.01

    def test_cache_hit_rate(self):
        t = PerformanceTracker()
        t.record("Q1", 0.01, cache_hit=True)
        t.record("Q2", 2.0, cache_hit=False)
        t.record("Q3", 0.02, cache_hit=True)
        assert abs(t.cache_hit_rate - 2 / 3) < 0.01

    def test_error_rate(self):
        t = PerformanceTracker()
        t.record("Q1", 1.0)
        t.record("Q2", 5.0, error="Timeout")
        assert abs(t.error_rate - 0.5) < 0.01

    def test_p95_latency(self):
        t = PerformanceTracker()
        for i in range(20):
            t.record(f"Q{i}", float(i))
        # p95 should be near 19
        assert t.p95_latency >= 18.0

    def test_summary(self):
        t = PerformanceTracker()
        t.record("Q1", 1.5, confidence=0.85)
        s = t.summary()
        assert s["total_queries"] == 1
        assert s["avg_latency_s"] == 1.5
        assert s["avg_confidence"] == 0.85

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "perf.json"
            t = PerformanceTracker()
            t.record("Q1", 1.0, confidence=0.8, source_count=3)
            t.record("Q2", 2.0, cache_hit=True)
            t.save(path)

            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["summary"]["total_queries"] == 2
            assert len(data["records"]) == 2

            # Load back
            t2 = PerformanceTracker.load(path)
            assert t2.total_queries == 2
            assert t2.records[0].question == "Q1"
            assert t2.records[1].cache_hit

    def test_load_nonexistent(self):
        t = PerformanceTracker.load("/nonexistent/path.json")
        assert t.total_queries == 0

    def test_save_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "perf.json"
            t = PerformanceTracker()
            t.record("Q1", 1.0)
            t.save(path)
            assert path.exists()
