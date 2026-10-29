"""Offline regressions for adapter failures, async retries and prompt loading."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.rag_adapter import RAGAdapter
from src.core.config import settings
from src.core.prompts import Prompts
from src.utils.robustness import retry_with_backoff


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
