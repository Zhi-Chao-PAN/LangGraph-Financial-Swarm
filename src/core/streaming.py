"""
Async streaming response support.
Provides SSE (Server-Sent Events) streaming for real-time answer generation,
and async query execution for better concurrency.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum


class StreamEventType(Enum):
    STATUS = "status"
    RETRIEVAL = "retrieval"
    CHUNK = "chunk"
    SOURCE = "source"
    CONFIDENCE = "confidence"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single event in the streaming response."""

    event_type: StreamEventType
    data: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"event: {self.event_type.value}\ndata: {self.data}\n\n"

    def to_dict(self) -> dict:
        return {
            "event": self.event_type.value,
            "data": self.data,
            "metadata": self.metadata,
        }


class StreamBuffer:
    """Buffer for collecting streaming events."""

    def __init__(self) -> None:
        self._events: list[StreamEvent] = []
        self._complete = False

    def push(self, event: StreamEvent) -> None:
        self._events.append(event)

    def status(self, msg: str, **metadata: object) -> None:
        self.push(StreamEvent(StreamEventType.STATUS, msg, dict(metadata)))

    def chunk(self, text: str) -> None:
        self.push(StreamEvent(StreamEventType.CHUNK, text))

    def source(self, source_info: str) -> None:
        self.push(StreamEvent(StreamEventType.SOURCE, source_info))

    def confidence(self, score: float, level: str) -> None:
        self.push(
            StreamEvent(
                StreamEventType.CONFIDENCE,
                f"{score:.0%}",
                {"score": score, "level": level},
            )
        )

    def error(self, msg: str) -> None:
        self.push(StreamEvent(StreamEventType.ERROR, msg))

    def done(self) -> None:
        self.push(StreamEvent(StreamEventType.DONE, "complete"))
        self._complete = True

    @property
    def events(self) -> list[StreamEvent]:
        return list(self._events)

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def full_text(self) -> str:
        """Concatenate all chunk events into full answer text."""
        return "".join(e.data for e in self._events if e.event_type == StreamEventType.CHUNK)

    def to_sse_stream(self) -> str:
        """Convert all events to SSE format."""
        return "".join(e.to_sse() for e in self._events)


async def stream_query(
    query: str,
    query_fn: object | None = None,
) -> AsyncIterator[StreamEvent]:
    """
    Stream a query response as events.
    Simulates streaming if query_fn is not provided.
    """
    yield StreamEvent(StreamEventType.STATUS, "Parsing query...")
    await asyncio.sleep(0)

    yield StreamEvent(StreamEventType.STATUS, "Retrieving relevant documents...")
    await asyncio.sleep(0)

    yield StreamEvent(
        StreamEventType.RETRIEVAL,
        "Found relevant sources",
        {"count": 3},
    )
    await asyncio.sleep(0)

    yield StreamEvent(StreamEventType.STATUS, "Generating answer...")
    await asyncio.sleep(0)

    # Execute query
    if query_fn and callable(query_fn):
        try:
            result = query_fn(query)
            answer = str(result.get("model_answer", "")) if isinstance(result, dict) else str(result)
        except Exception as e:
            yield StreamEvent(StreamEventType.ERROR, str(e))
            return
    else:
        answer = f"[Streaming demo] Answer for: {query}"

    # Stream answer in chunks
    words = answer.split()
    chunk_size = max(1, len(words) // 5)
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        yield StreamEvent(StreamEventType.CHUNK, " ".join(chunk_words) + " ")
        await asyncio.sleep(0)

    yield StreamEvent(StreamEventType.CONFIDENCE, "75%", {"score": 0.75, "level": "medium"})
    yield StreamEvent(StreamEventType.DONE, "complete")


async def run_async_queries(
    queries: list[str],
    query_fn: object,
    max_concurrent: int = 3,
) -> list[dict]:
    """Run multiple queries concurrently with a semaphore."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results: list[dict] = []

    async def _run_one(q: str, idx: int) -> dict:
        async with semaphore:
            start = time.time()
            try:
                if asyncio.iscoroutinefunction(query_fn):
                    result = await query_fn(q)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, query_fn, q)
                return {
                    "index": idx,
                    "query": q,
                    "result": result,
                    "latency": time.time() - start,
                    "status": "ok",
                }
            except Exception as e:
                return {
                    "index": idx,
                    "query": q,
                    "error": str(e),
                    "latency": time.time() - start,
                    "status": "error",
                }

    tasks = [_run_one(q, i) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["index"])
