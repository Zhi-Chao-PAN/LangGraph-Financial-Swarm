"""Tests for observability and tracing."""

from src.utils.observability import Observability


class TestObservability:
    def test_start_trace(self):
        qid = Observability.start_trace("Test query")
        assert len(qid) == 8
        assert Observability._query_id == qid

    def test_trace_agent(self, capsys):
        Observability.start_trace("Test")
        Observability.trace_agent("Researcher", "Found revenue data")
        assert Observability._agent_calls["Researcher"] == 1

    def test_trace_multiple_agents(self):
        Observability.start_trace("Test")
        Observability.trace_agent("Supervisor", "Routing to Researcher")
        Observability.trace_agent("Researcher", "Querying RAG")
        Observability.trace_agent("Researcher", "Got answer")
        assert Observability._agent_calls["Supervisor"] == 1
        assert Observability._agent_calls["Researcher"] == 2

    def test_trace_error(self):
        Observability.start_trace("Test")
        Observability.trace_error("Researcher", "Timeout", recoverable=True)
        assert len(Observability._errors) == 1
        assert Observability._errors[0]["recoverable"]

    def test_trace_error_unrecoverable(self):
        Observability.start_trace("Test")
        Observability.trace_error("Supervisor", "Fatal crash", recoverable=False)
        assert not Observability._errors[0]["recoverable"]

    def test_final_report(self, capsys):
        Observability.start_trace("Test")
        Observability.trace_agent("Researcher", "Answer")
        Observability.final_report(total_steps=1)

    def test_get_metrics(self):
        Observability.start_trace("Test")
        Observability.trace_agent("Researcher", "Data")
        metrics = Observability.get_metrics()
        assert metrics["query_id"] == Observability._query_id
        assert metrics["agent_calls"]["Researcher"] == 1
        assert metrics["error_count"] == 0
        assert metrics["elapsed_s"] >= 0

    def test_trace_tool(self, capsys):
        Observability.start_trace("Test")
        Observability.trace_tool("create_plot", '{"type": "bar"}', "Chart generated")

    def test_agent_latencies_tracked(self):
        Observability.start_trace("Test")
        Observability.trace_agent("Researcher", "Step 1")
        Observability.trace_agent("Researcher", "Step 2")
        assert len(Observability._agent_latencies["Researcher"]) == 2
