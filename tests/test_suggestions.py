"""Tests for smart search suggestions."""

import json
import tempfile
from pathlib import Path

from src.core.suggestions import DEFAULT_SUGGESTIONS, SearchSuggester


class TestSearchSuggester:
    def test_insert_and_suggest(self):
        s = SearchSuggester()
        s.insert("What was NVIDIA revenue?")
        s.insert("What was Apple revenue?")
        results = s.suggest("what was")
        assert len(results) == 2

    def test_suggest_empty_prefix(self):
        s = SearchSuggester()
        s.insert("NVIDIA revenue")
        s.insert("NVIDIA revenue")
        s.insert("Apple income")
        results = s.suggest("")
        assert results[0] == "NVIDIA revenue"  # Higher count

    def test_suggest_no_match(self):
        s = SearchSuggester()
        s.insert("NVIDIA revenue")
        results = s.suggest("zzz")
        assert results == []

    def test_frequency_ranking(self):
        s = SearchSuggester()
        s.insert("Apple revenue")
        s.insert("NVIDIA revenue")
        s.insert("NVIDIA revenue")
        s.insert("NVIDIA revenue")
        results = s.suggest("", limit=2)
        assert results[0] == "NVIDIA revenue"

    def test_limit(self):
        s = SearchSuggester()
        for i in range(20):
            s.insert(f"query {i}")
        results = s.suggest("query", limit=5)
        assert len(results) == 5

    def test_case_insensitive(self):
        s = SearchSuggester()
        s.insert("NVIDIA Revenue")
        results = s.suggest("nvidia")
        assert len(results) == 1

    def test_empty_insert(self):
        s = SearchSuggester()
        s.insert("")
        s.insert("  ")
        assert s.unique_queries == 0

    def test_chinese_queries(self):
        s = SearchSuggester()
        s.insert("英伟达的营收是多少？")
        s.insert("英伟达的净利润")
        results = s.suggest("英伟达")
        assert len(results) == 2

    def test_hot_queries(self):
        s = SearchSuggester()
        s.insert("q1")
        s.insert("q2")
        s.insert("q2")
        s.insert("q3")
        s.insert("q3")
        s.insert("q3")
        hot = s.hot_queries(limit=2)
        assert hot[0] == "q3"
        assert hot[1] == "q2"

    def test_recent_queries(self):
        s = SearchSuggester()
        s.insert("old query")
        # Manually set older timestamp
        s._stats["old query"].last_used = 1000
        s.insert("new query")
        s._stats["new query"].last_used = 2000
        recent = s.recent_queries(limit=1)
        assert recent[0] == "new query"

    def test_total_and_unique(self):
        s = SearchSuggester()
        s.insert("q1")
        s.insert("q1")
        s.insert("q2")
        assert s.total_queries == 3
        assert s.unique_queries == 2

    def test_max_history_eviction(self):
        s = SearchSuggester(max_history=5)
        for i in range(10):
            s.insert(f"query {i}")
        assert s.unique_queries <= 5

    def test_save_and_load(self):
        s = SearchSuggester()
        s.insert("NVIDIA revenue")
        s.insert("NVIDIA revenue")
        s.insert("Apple income")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        s.save(path)

        # Verify JSON
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "stats" in data

        # Load into new instance
        s2 = SearchSuggester()
        s2.load(path)
        assert s2.unique_queries == 2
        assert s2.total_queries == 3
        results = s2.suggest("nvidia")
        assert len(results) == 1

        Path(path).unlink()

    def test_load_nonexistent(self):
        s = SearchSuggester()
        s.load("/nonexistent/path.json")
        assert s.unique_queries == 0


class TestDefaultSuggestions:
    def test_defaults_exist(self):
        assert len(DEFAULT_SUGGESTIONS) >= 5

    def test_has_chinese(self):
        chinese = [s for s in DEFAULT_SUGGESTIONS if any("\u4e00" <= c <= "\u9fff" for c in s)]
        assert len(chinese) >= 1

    def test_has_english(self):
        english = [s for s in DEFAULT_SUGGESTIONS if s[0].isascii()]
        assert len(english) >= 3
