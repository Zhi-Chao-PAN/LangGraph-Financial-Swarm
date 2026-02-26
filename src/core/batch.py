"""
Batch query executor for systematic evaluation and data collection.
Supports running queries from templates, files, or programmatic generation.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.robustness import log_agent_action


@dataclass
class BatchResult:
    """Result of a single query in a batch."""

    question: str
    answer: str
    latency_s: float
    confidence: float | None = None
    source_count: int = 0
    error: str | None = None


@dataclass
class BatchReport:
    """Summary report for a batch execution."""

    results: list[BatchResult] = field(default_factory=list)
    total_time_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if not r.error)

    @property
    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)

    @property
    def avg_latency(self) -> float:
        successful = [r.latency_s for r in self.results if not r.error]
        return sum(successful) / len(successful) if successful else 0.0

    @property
    def avg_confidence(self) -> float:
        confident = [r.confidence for r in self.results if r.confidence is not None]
        return sum(confident) / len(confident) if confident else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "success": self.success_count,
            "errors": self.error_count,
            "avg_latency_s": round(self.avg_latency, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "total_time_s": round(self.total_time_s, 3),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "results": [asdict(r) for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log_agent_action("Batch", "Saved", f"{path}: {self.total} results")


def run_batch(
    queries: list[str],
    adapter=None,
    progress_callback=None,
) -> BatchReport:
    """
    Execute a batch of queries sequentially.

    Args:
        queries: List of questions to execute
        adapter: RAG adapter (uses global if None)
        progress_callback: Optional callback(i, total, question) for progress updates
    """
    if adapter is None:
        from src.rag_adapter import adapter as default_adapter

        adapter = default_adapter

    report = BatchReport()
    start = time.time()

    log_agent_action("Batch", "Start", f"{len(queries)} queries")

    for i, question in enumerate(queries):
        if progress_callback:
            progress_callback(i, len(queries), question)

        q_start = time.time()
        try:
            result = adapter.query(question)
            latency = time.time() - q_start
            answer = result.get("model_answer", "")

            # Extract confidence if present
            import re

            confidence = None
            match = re.search(r"\[Confidence:\s*([\d.]+)\]", answer)
            if match:
                confidence = float(match.group(1))

            report.results.append(
                BatchResult(
                    question=question,
                    answer=answer,
                    latency_s=latency,
                    confidence=confidence,
                    source_count=len(result.get("source_nodes", [])),
                )
            )
        except Exception as e:
            report.results.append(
                BatchResult(
                    question=question,
                    answer="",
                    latency_s=time.time() - q_start,
                    error=str(e),
                )
            )
            log_agent_action("Batch", "Error", f"Q{i}: {e}", level="warning")

    report.total_time_s = time.time() - start
    log_agent_action("Batch", "Done", f"{report.success_count}/{report.total} in {report.total_time_s:.1f}s")
    return report


def load_queries_from_file(path: str | Path) -> list[str]:
    """Load queries from a text file (one per line) or JSON array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    if path.suffix == ".json":
        data = json.loads(content)
        if isinstance(data, list):
            return [str(q) for q in data]
        raise ValueError("JSON file must contain a list of queries")

    # Text file: one query per line
    return [line.strip() for line in content.split("\n") if line.strip() and not line.startswith("#")]
