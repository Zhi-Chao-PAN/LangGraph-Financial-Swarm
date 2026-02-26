"""Tests for async streaming response support."""

import asyncio

import pytest

from src.core.streaming import (
    StreamBuffer,
    StreamEvent,
    StreamEventType,
    run_async_queries,
    stream_query,
)


class TestStreamEvent:
    def test_basic(self):
        e = StreamEvent(StreamEventType.CHUNK, "hello")
        assert e.event_type == StreamEventType.CHUNK
        assert e.data == "hello"

    def test_to_sse(self):
        e = StreamEvent(StreamEventType.STATUS, "loading")
        sse = e.to_sse()
        assert "event: status" in sse
        assert "data: loading" in sse

    def test_to_dict(self):
        e = StreamEvent(StreamEventType.CONFIDENCE, "85%", {"score": 0.85})
        d = e.to_dict()
        assert d["event"] == "confidence"
        assert d["metadata"]["score"] == 0.85


class TestStreamBuffer:
    def test_push(self):
        buf = StreamBuffer()
        buf.push(StreamEvent(StreamEventType.CHUNK, "hello"))
        assert len(buf.events) == 1

    def test_status(self):
        buf = StreamBuffer()
        buf.status("loading", step=1)
        assert buf.events[0].event_type == StreamEventType.STATUS
        assert buf.events[0].metadata["step"] == 1

    def test_chunk(self):
        buf = StreamBuffer()
        buf.chunk("part1 ")
        buf.chunk("part2")
        assert buf.full_text == "part1 part2"

    def test_source(self):
        buf = StreamBuffer()
        buf.source("NVIDIA FY2025")
        assert buf.events[0].event_type == StreamEventType.SOURCE

    def test_confidence(self):
        buf = StreamBuffer()
        buf.confidence(0.85, "high")
        assert buf.events[0].metadata["score"] == 0.85

    def test_error(self):
        buf = StreamBuffer()
        buf.error("timeout")
        assert buf.events[0].event_type == StreamEventType.ERROR

    def test_done(self):
        buf = StreamBuffer()
        assert not buf.is_complete
        buf.done()
        assert buf.is_complete

    def test_full_text_only_chunks(self):
        buf = StreamBuffer()
        buf.status("loading")
        buf.chunk("hello ")
        buf.source("src1")
        buf.chunk("world")
        buf.done()
        assert buf.full_text == "hello world"

    def test_to_sse_stream(self):
        buf = StreamBuffer()
        buf.status("start")
        buf.chunk("data")
        buf.done()
        sse = buf.to_sse_stream()
        assert "event: status" in sse
        assert "event: chunk" in sse
        assert "event: done" in sse


class TestStreamQuery:
    @pytest.mark.asyncio
    async def test_stream_demo(self):
        events = []
        async for event in stream_query("What was NVIDIA revenue?"):
            events.append(event)
        types = [e.event_type for e in events]
        assert StreamEventType.STATUS in types
        assert StreamEventType.CHUNK in types
        assert StreamEventType.DONE in types

    @pytest.mark.asyncio
    async def test_stream_with_fn(self):
        def mock_fn(q):
            return {"model_answer": f"Answer for {q}"}

        events = []
        async for event in stream_query("test query", mock_fn):
            events.append(event)
        chunks = [e.data for e in events if e.event_type == StreamEventType.CHUNK]
        full = "".join(chunks)
        assert "Answer" in full

    @pytest.mark.asyncio
    async def test_stream_error(self):
        def bad_fn(q):
            raise ValueError("LLM timeout")

        events = []
        async for event in stream_query("test", bad_fn):
            events.append(event)
        errors = [e for e in events if e.event_type == StreamEventType.ERROR]
        assert len(errors) == 1
        assert "timeout" in errors[0].data


class TestRunAsyncQueries:
    @pytest.mark.asyncio
    async def test_concurrent(self):
        def mock_fn(q):
            return {"answer": q}

        results = await run_async_queries(
            ["q1", "q2", "q3"],
            mock_fn,
            max_concurrent=2,
        )
        assert len(results) == 3
        assert results[0]["index"] == 0
        assert results[2]["index"] == 2
        assert all(r["status"] == "ok" for r in results)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        def bad_fn(q):
            if "bad" in q:
                raise ValueError("fail")
            return {"answer": q}

        results = await run_async_queries(
            ["good", "bad query"],
            bad_fn,
        )
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_async_fn(self):
        async def async_fn(q):
            await asyncio.sleep(0)
            return {"answer": q}

        results = await run_async_queries(["q1"], async_fn)
        assert len(results) == 1
        assert results[0]["status"] == "ok"
