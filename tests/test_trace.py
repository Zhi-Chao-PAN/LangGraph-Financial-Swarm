"""Tests for agent reasoning trace visualization."""

from src.utils.trace import (
    ReasoningTrace,
    StepType,
    TraceStep,
)


class TestTraceStep:
    def test_basic(self):
        step = TraceStep(
            step_type=StepType.RETRIEVAL,
            agent="researcher",
            description="Search for NVIDIA revenue",
            duration_ms=150.5,
        )
        assert step.agent == "researcher"
        assert step.step_type == StepType.RETRIEVAL

    def test_to_dict(self):
        step = TraceStep(
            step_type=StepType.LLM_CALL,
            agent="synthesizer",
            description="Generate answer",
            input_data="context here",
            output_data="Revenue was $130B",
            duration_ms=2000,
        )
        d = step.to_dict()
        assert d["type"] == "llm_call"
        assert d["agent"] == "synthesizer"
        assert d["duration_ms"] == 2000.0

    def test_long_data_truncated(self):
        step = TraceStep(
            step_type=StepType.RETRIEVAL,
            agent="test",
            description="test",
            input_data="x" * 1000,
            output_data="y" * 1000,
        )
        d = step.to_dict()
        assert len(d["input"]) == 500
        assert len(d["output"]) == 500


class TestReasoningTrace:
    def _make_trace(self) -> ReasoningTrace:
        trace = ReasoningTrace(query="What was NVIDIA revenue?")
        trace.add_step(
            StepType.QUERY_PARSE,
            "router",
            "Parse query",
            input_data="What was NVIDIA revenue?",
            output_data="company=NVIDIA, metric=revenue",
            duration_ms=5,
        )
        trace.add_step(
            StepType.SUB_QUERY_GEN,
            "router",
            "Generate sub-queries",
            output_data="['NVIDIA revenue FY2025']",
            duration_ms=10,
        )
        trace.add_step(
            StepType.RETRIEVAL,
            "researcher",
            "RAG retrieval",
            output_data="Found 3 relevant chunks",
            duration_ms=150,
            top_k=5,
        )
        trace.add_step(
            StepType.LLM_CALL,
            "synthesizer",
            "Generate answer",
            duration_ms=2000,
        )
        trace.add_step(
            StepType.QUALITY_CHECK,
            "quality",
            "Validate answer",
            duration_ms=5,
        )
        trace.finish()
        return trace

    def test_add_step(self):
        trace = ReasoningTrace(query="test")
        trace.add_step(StepType.RETRIEVAL, "agent", "desc")
        assert len(trace.steps) == 1

    def test_finish(self):
        trace = self._make_trace()
        assert trace.total_duration_ms >= 0

    def test_to_dict(self):
        trace = self._make_trace()
        d = trace.to_dict()
        assert d["query"] == "What was NVIDIA revenue?"
        assert d["step_count"] == 5
        assert len(d["steps"]) == 5

    def test_to_markdown(self):
        trace = self._make_trace()
        md = trace.to_markdown()
        assert "Reasoning Trace" in md
        assert "NVIDIA" in md
        assert "Step 1" in md
        assert "Step 5" in md
        assert "router" in md
        assert "researcher" in md

    def test_to_mermaid(self):
        trace = self._make_trace()
        mermaid = trace.to_mermaid()
        assert "graph TD" in mermaid
        assert "S0" in mermaid
        assert "S4" in mermaid
        assert "-->" in mermaid
        assert "fill:" in mermaid

    def test_agent_summary(self):
        trace = self._make_trace()
        summary = trace.agent_summary
        assert "router" in summary
        assert "researcher" in summary
        assert summary["router"] == 15  # 5 + 10

    def test_metadata(self):
        trace = ReasoningTrace(query="test")
        step = trace.add_step(
            StepType.RETRIEVAL,
            "researcher",
            "search",
            top_k=5,
            method="hybrid",
        )
        assert step.metadata["top_k"] == 5
        assert step.metadata["method"] == "hybrid"

    def test_empty_trace(self):
        trace = ReasoningTrace(query="empty")
        trace.finish()
        assert len(trace.steps) == 0
        md = trace.to_markdown()
        assert "Steps: 0" in md

    def test_error_step(self):
        trace = ReasoningTrace(query="test")
        trace.add_step(
            StepType.ERROR,
            "synthesizer",
            "LLM timeout",
            duration_ms=30000,
        )
        mermaid = trace.to_mermaid()
        assert "{{" in mermaid  # Error shape

    def test_step_icons_in_markdown(self):
        trace = ReasoningTrace(query="test")
        trace.add_step(StepType.RETRIEVAL, "r", "search")
        trace.add_step(StepType.LLM_CALL, "s", "generate")
        trace.add_step(StepType.ERROR, "e", "fail")
        md = trace.to_markdown()
        assert "🔎" in md
        assert "🤖" in md
        assert "❌" in md
