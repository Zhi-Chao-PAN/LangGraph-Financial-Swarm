"""Offline regressions for adapter failures, async retries and prompt loading."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.rag_adapter import RAGAdapter
from src.core.config import settings
from src.core.prompts import Prompts
from src.utils.robustness import retry_with_backoff
from pathlib import Path


@pytest.mark.asyncio
async def test_rag_network_failure() -> None:
    with (
        patch("src.rag_adapter.dc.Cache") as cache_factory,
        patch("src.utils.robustness.asyncio.sleep", new_callable=AsyncMock),
    ):
        cache_factory.return_value.get.return_value = None
        adapter = RAGAdapter()
        engine = MagicMock()
        engine.aquery = AsyncMock(side_effect=ConnectionError("Connection Timeout"))
        adapter.query_engine = engine
        result = await adapter.aquery("What is NVIDIA's revenue?")
        assert result["model_answer"].startswith("Error: RAG Engine Connection Failed:")
        assert result["source_nodes"] == []
        assert result["latency_s"] == 0.0
        assert engine.aquery.await_count == 4
        cache_factory.return_value.set.assert_not_called()


@pytest.mark.asyncio
async def test_async_retry_awaits_transient_failures() -> None:
    operation = AsyncMock(side_effect=[ConnectionError("temporary"), "recovered"])

    @retry_with_backoff(retries=1, backoff_in_seconds=0)
    async def call() -> str:
        return str(await operation())

    assert await call() == "recovered"
    assert operation.await_count == 2


@pytest.mark.asyncio
async def test_retry_does_not_swallow_cancellation() -> None:
    @retry_with_backoff(retries=2, backoff_in_seconds=0)
    async def cancelled() -> str:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cancelled()


def test_sync_retry_preserves_result() -> None:
    operation = MagicMock(side_effect=[ValueError("temporary"), 42])

    @retry_with_backoff(retries=1, backoff_in_seconds=0)
    def call() -> int:
        return int(operation())

    assert call() == 42
    assert operation.call_count == 2


@pytest.mark.asyncio
async def test_sync_adapter_rejects_nested_event_loop() -> None:
    with patch("src.rag_adapter.dc.Cache"):
        adapter = RAGAdapter()
        with pytest.raises(RuntimeError, match="aquery"):
            adapter.query("question")


def test_config_load() -> None:
    assert settings.LLM_MODEL
    assert settings.LLM_TIMEOUT > 0


def test_prompts_are_loaded_from_bundled_files() -> None:
    assert Prompts.SUPERVISOR_SYSTEM == (Prompts.BASE_DIR / "supervisor.txt").read_text(
        encoding="utf-8"
    )
    assert Prompts.RESEARCHER_SYSTEM == (Prompts.BASE_DIR / "researcher.txt").read_text(
        encoding="utf-8"
    )
    assert Prompts.QUANT_SYSTEM == (Prompts.BASE_DIR / "quant.txt").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "set"])
async def test_cache_failure_does_not_discard_a_successful_answer(
    operation: str,
) -> None:
    with patch("src.rag_adapter.dc.Cache") as factory:
        factory.return_value.get.return_value = None
        getattr(factory.return_value, operation).side_effect = OSError(
            "cache unavailable"
        )
        adapter = RAGAdapter()
        engine = MagicMock()
        engine.aquery = AsyncMock(return_value="42")
        adapter.query_engine = engine
        result = await adapter.aquery("question")
    assert result["model_answer"].startswith("42")
    engine.aquery.assert_awaited_once()


def test_missing_corpus_fails_before_embedding_initialization(tmp_path: Path) -> None:
    with (
        patch("src.rag_adapter.dc.Cache"),
        patch.object(settings, "RAG_DATA_PATH", tmp_path / "missing.md"),
        patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding") as embedding,
    ):
        with pytest.raises(FileNotFoundError, match="RAG data not found"):
            RAGAdapter()._initialize_sync()
        embedding.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_initialization_builds_once_and_honors_top_k() -> None:
    with patch("src.rag_adapter.dc.Cache"), patch("src.rag_adapter.Ollama"):
        adapter = RAGAdapter()
        index = MagicMock()
        with (
            patch.object(adapter, "_initialize_sync", return_value=index) as build,
            patch.object(settings, "RAG_TOP_K", 7),
        ):
            await asyncio.gather(
                *(adapter._ensure_initialized_async() for _ in range(8))
            )
        build.assert_called_once()
        assert index.as_query_engine.call_args.kwargs["similarity_top_k"] == 7


@pytest.mark.asyncio
async def test_partial_engine_initialization_can_retry() -> None:
    with patch("src.rag_adapter.dc.Cache"), patch("src.rag_adapter.Ollama"):
        adapter = RAGAdapter()
        index = MagicMock()
        engine = MagicMock()
        index.as_query_engine.side_effect = [
            RuntimeError("temporary setup error"),
            engine,
        ]
        with patch.object(adapter, "_initialize_sync", return_value=index):
            with pytest.raises(RuntimeError, match="temporary setup error"):
                await adapter._ensure_initialized_async()
            await adapter._ensure_initialized_async()
        assert adapter.query_engine is engine
