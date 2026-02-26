"""
Smart search suggestions with Trie prefix tree and hot query tracking.
Provides autocomplete and popular query recommendations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QueryStats:
    """Track query frequency and recency."""

    query: str
    count: int = 0
    last_used: float = 0.0

    def to_dict(self) -> dict:
        return {"query": self.query, "count": self.count, "last_used": self.last_used}


class TrieNode:
    """Node in the prefix tree."""

    __slots__ = ("children", "is_end", "query")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.query: str = ""


class SearchSuggester:
    """
    Smart search suggestion engine.
    Combines Trie-based prefix matching with hot query tracking.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._root = TrieNode()
        self._stats: dict[str, QueryStats] = {}
        self._max_history = max_history

    def insert(self, query: str) -> None:
        """Insert a query into the Trie and update stats."""
        normalized = query.strip().lower()
        if not normalized:
            return

        # Update Trie
        node = self._root
        for char in normalized:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.query = query.strip()

        # Update stats
        if normalized not in self._stats:
            self._stats[normalized] = QueryStats(query=query.strip())
        self._stats[normalized].count += 1
        self._stats[normalized].last_used = time.time()

        # Evict old entries if over limit
        if len(self._stats) > self._max_history:
            oldest_key = min(self._stats, key=lambda k: self._stats[k].last_used)
            del self._stats[oldest_key]

    def suggest(self, prefix: str, limit: int = 5) -> list[str]:
        """Get autocomplete suggestions for a prefix."""
        normalized = prefix.strip().lower()
        if not normalized:
            return self.hot_queries(limit)

        # Navigate to prefix node
        node = self._root
        for char in normalized:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all completions
        results: list[str] = []
        self._collect(node, results, limit * 3)

        # Sort by frequency
        scored = []
        for q in results:
            key = q.lower()
            count = self._stats[key].count if key in self._stats else 0
            scored.append((q, count))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [q for q, _ in scored[:limit]]

    def _collect(self, node: TrieNode, results: list[str], limit: int) -> None:
        """DFS to collect all queries under a node."""
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(node.query)
        for child in node.children.values():
            self._collect(child, results, limit)

    def hot_queries(self, limit: int = 10) -> list[str]:
        """Get the most popular queries."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: (s.count, s.last_used),
            reverse=True,
        )
        return [s.query for s in sorted_stats[:limit]]

    def recent_queries(self, limit: int = 10) -> list[str]:
        """Get the most recent queries."""
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.last_used,
            reverse=True,
        )
        return [s.query for s in sorted_stats[:limit]]

    @property
    def total_queries(self) -> int:
        return sum(s.count for s in self._stats.values())

    @property
    def unique_queries(self) -> int:
        return len(self._stats)

    def save(self, path: str | Path) -> None:
        """Save stats to JSON."""
        data = {
            "stats": {k: v.to_dict() for k, v in self._stats.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Load stats from JSON and rebuild Trie."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        for _key, stat in data.get("stats", {}).items():
            q = stat["query"]
            # Rebuild Trie
            normalized = q.strip().lower()
            node = self._root
            for char in normalized:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_end = True
            node.query = q
            # Restore stats
            self._stats[normalized] = QueryStats(
                query=q,
                count=stat["count"],
                last_used=stat.get("last_used", 0),
            )


# Default templates for cold start
DEFAULT_SUGGESTIONS = [
    "What was NVIDIA revenue in FY2025?",
    "Compare Apple and Microsoft net income",
    "NVIDIA gross margin trend",
    "Amazon free cash flow",
    "Alphabet operating income FY2024",
    "英伟达2025财年营收是多少？",
    "对比苹果和微软的净利润",
    "亚马逊的自由现金流",
]
