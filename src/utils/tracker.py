"""
Query performance tracker for monitoring and analysis.
Records latency, cache hits, confidence, and source counts per query.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QueryRecord:
    """A single query performance record."""

    question: str
    latency_s: float
    confidence: float | None = None
    source_count: int = 0
    cache_hit: bool = False
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceTracker:
    """Tracks query performance metrics across a session."""

    records: list[QueryRecord] = field(default_factory=list)

    def record(self, question: str, latency_s: float, **kwargs: Any) -> QueryRecord:
        """Record a query result."""
        rec = QueryRecord(question=question, latency_s=latency_s, **kwargs)
        self.records.append(rec)
        return rec

    @property
    def total_queries(self) -> int:
        return len(self.records)

    @property
    def avg_latency(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_s for r in self.records) / len(self.records)

    @property
    def p95_latency(self) -> float:
        if not self.records:
            return 0.0
        sorted_latencies = sorted(r.latency_s for r in self.records)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def cache_hit_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.cache_hit) / len(self.records)

    @property
    def avg_confidence(self) -> float:
        confident = [r.confidence for r in self.records if r.confidence is not None]
        if not confident:
            return 0.0
        return sum(confident) / len(confident)

    @property
    def error_rate(self) -> float:
        if not self.records:
            return 0.0
        return sum(1 for r in self.records if r.error) / len(self.records)

    def summary(self) -> dict[str, Any]:
        """Get a summary of all tracked metrics."""
        return {
            "total_queries": self.total_queries,
            "avg_latency_s": round(self.avg_latency, 3),
            "p95_latency_s": round(self.p95_latency, 3),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "error_rate": round(self.error_rate, 3),
        }

    def save(self, path: str | Path) -> None:
        """Save all records to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "PerformanceTracker":
        """Load records from JSON."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tracker = cls()
        for rec_data in data.get("records", []):
            tracker.records.append(QueryRecord(**rec_data))
        return tracker


# Global tracker instance
tracker = PerformanceTracker()
