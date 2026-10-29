"""Exercise the production graph with scripted models and offline tools."""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from src.core.config import settings
from src.core.demo import DEMO_ROWS, run_demo
from src.core.workflow import build_swarm_graph
from src.utils.tool_parsing import ToolParser
from src.utils.validation import sanitize_input


@pytest.mark.asyncio
async def test_offline_demo_runs_real_graph_and_exports_auditable_chart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "OUTPUT_DIR", tmp_path)
    result = await run_demo()
    assert result["synthetic"] is True
    assert result["model_backend"] == "scripted"
    assert result["benchmark_claim"] == "none"
    assert result["steps"] == [
        "Supervisor",
        "Researcher",
        "tools",
        "Researcher",
        "Supervisor",
        "Quant",
        "tools",
        "Quant",
        "Supervisor",
    ]
    assert [entry["name"] for entry in result["tools"]] == [
        "query_financial_rag",
        "create_plot",
    ]
    assert "20%" in result["answer"]
    assert json.loads(result["tools"][0]["result"])["rows"] == DEMO_ROWS
    charts = list(tmp_path.glob("*.png"))
    csvs = list(tmp_path.glob("*.csv"))
    assert len(charts) == len(csvs) == 1
    assert charts[0].read_bytes().startswith(b"\x89PNG")
    assert csvs[0].read_text().splitlines() == ["Year,Revenue", "2023,100", "2024,120"]


@pytest.mark.asyncio
async def test_demo_honors_an_explicit_step_budget() -> None:
    with pytest.raises(GraphRecursionError):
        await run_demo(max_steps=1)


@pytest.mark.asyncio
async def test_unfinished_workflows_respect_the_recursion_budget() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="Next: Researcher"),
            AIMessage(content="Need more information."),
        ]
    )
    graph = build_swarm_graph(model, tools=[])
    with pytest.raises(GraphRecursionError):
        await graph.ainvoke(
            {"messages": [("user", "question")]}, {"recursion_limit": 4}
        )


@pytest.mark.parametrize(
    "args",
    [
        {"nested": {"rows": [{"year": 2024, "value": 120}]}},
        {"data_str": '[{"label":"brace } and quote \\"","value":120}]'},
        {"question": "比较收入（人民币 ¥）"},
    ],
)
def test_tool_arguments_preserve_nested_json_and_quoted_braces(args: dict) -> None:
    call = ToolParser.parse_tool_call(
        f"TOOL_CALL: QUERY_FINANCIAL_RAG\nARGS: {json.dumps(args)}\nExplanation follows.",
        ["query_financial_rag"],
        "Researcher",
    )
    assert call is not None
    assert call["name"] == "query_financial_rag"
    assert call["args"] == args


@pytest.mark.parametrize(
    "content",
    [
        'TOOL_CALL: unknown\nARGS: {"question":"x"}',
        'TOOL_CALL: query_financial_rag\nARGS: ["not", "an", "object"]',
        'TOOL_CALL: query_financial_rag\nARGS: {"question":"unfinished',
        "No tool requested.",
    ],
)
def test_untrusted_or_incomplete_tool_calls_are_not_executed(content: str) -> None:
    assert (
        ToolParser.parse_tool_call(content, ["query_financial_rag"], "Researcher")
        is None
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("  比较英伟达 2023—2024 年收入 📈  ", "比较英伟达 2023—2024 年收入 📈"),
        ("إيرادات الشركة", "إيرادات الشركة"),
        ("\x00revenue\x1b\n2024", "revenue\n2024"),
        ("\x00\x01  ", ""),
        ("数" * 501, "数" * 500),
    ],
)
def test_query_normalization_preserves_unicode_and_bounds_input(
    query: str, expected: str
) -> None:
    assert sanitize_input(query) == expected
