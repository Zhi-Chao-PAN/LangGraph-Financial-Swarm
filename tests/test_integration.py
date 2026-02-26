import matplotlib

matplotlib.use("Agg")

import os
import subprocess
import sys
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from eval.run_eval import _approx_match, _extract_numbers, check_answer
from src.agents.researcher import _rewrite_query
from src.agents.supervisor import _keyword_route, _parse_routing, create_supervisor_node
from src.core.config import settings
from src.utils.validation import sanitize_input

# ─── Config ───


class TestConfig:
    def test_config_loading(self):
        assert settings.LLM_MODEL is not None
        assert settings.DATA_DIR is not None

    def test_dirs_exist(self):
        assert os.path.exists(settings.DATA_DIR)

    def test_rag_top_k(self):
        assert settings.RAG_TOP_K >= 3

    def test_llm_timeout(self):
        assert settings.LLM_TIMEOUT >= 60

    def test_validate_returns_list(self):
        warnings = settings.validate()
        assert isinstance(warnings, list)

    def test_validate_low_timeout(self):
        from src.core.config import Settings

        s = Settings(LLM_TIMEOUT=10)
        warnings = s.validate()
        assert any("timeout" in w.lower() for w in warnings)

    def test_validate_high_temperature(self):
        from src.core.config import Settings

        s = Settings(LLM_TEMPERATURE=0.9)
        warnings = s.validate()
        assert any("temperature" in w.lower() for w in warnings)

    def test_validate_low_top_k(self):
        from src.core.config import Settings

        s = Settings(RAG_TOP_K=1)
        warnings = s.validate()
        assert any("top_k" in w.lower() for w in warnings)


# ─── Supervisor Routing ───


class TestSupervisorRouting:
    def test_parse_next_researcher(self):
        assert _parse_routing("Next: Researcher", ["Researcher", "Quant"]) == "Researcher"

    def test_parse_next_quant(self):
        assert _parse_routing("Next: Quant", ["Researcher", "Quant"]) == "Quant"

    def test_parse_next_finish(self):
        assert _parse_routing("Next: FINISH", ["Researcher", "Quant"]) == "FINISH"

    def test_parse_case_insensitive(self):
        assert _parse_routing("next: researcher", ["Researcher", "Quant"]) == "Researcher"

    def test_parse_with_extra_text(self):
        content = "I think we need more data. Next: Researcher"
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Researcher"

    def test_parse_garbage_falls_back(self):
        content = "I am a stochastic parrot with no structured output."
        assert _parse_routing(content, ["Researcher", "Quant"]) == "Researcher"

    def test_parse_empty_falls_back(self):
        assert _parse_routing("", ["Researcher", "Quant"]) == "Researcher"

    def test_parse_standalone_word(self):
        assert _parse_routing("Quant", ["Researcher", "Quant"]) == "Quant"

    def test_parse_finish_standalone(self):
        assert _parse_routing("FINISH", ["Researcher", "Quant"]) == "FINISH"

    def test_keyword_route_chart(self):
        assert _keyword_route("Draw a bar chart of revenue", ["Researcher", "Quant"]) == "Quant"

    def test_keyword_route_plot(self):
        assert _keyword_route("Plot NVIDIA revenue trend", ["Researcher", "Quant"]) == "Quant"

    def test_keyword_route_visualize(self):
        assert _keyword_route("Visualize the margin comparison", ["Researcher", "Quant"]) == "Quant"

    def test_keyword_route_factual(self):
        assert _keyword_route("What was NVIDIA revenue in FY2025?", ["Researcher", "Quant"]) == "Researcher"

    def test_keyword_route_compare(self):
        assert _keyword_route("Compare Apple and Microsoft net income", ["Researcher", "Quant"]) == "Researcher"

    def test_supervisor_node_no_response(self):
        """No agent response yet -> route to Researcher."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Researcher")
        supervisor = create_supervisor_node(mock_llm, ["Researcher", "Quant"])
        result = supervisor({"messages": [("user", "What is NVIDIA revenue?")]})
        assert result["next"] in ["Researcher", "Quant", "FINISH"]

    def test_supervisor_node_after_response(self):
        """Agent already responded -> FINISH."""
        mock_llm = MagicMock()
        supervisor = create_supervisor_node(mock_llm, ["Researcher", "Quant"])
        result = supervisor(
            {
                "messages": [
                    ("user", "What is revenue?"),
                    AIMessage(content="Revenue is $130,497M. Citations: source1"),
                ]
            }
        )
        assert result["next"] == "FINISH"


# ─── Sanitize Input ───


class TestSanitizeInput:
    def test_basic(self):
        assert sanitize_input("  hello  ") == "hello"

    def test_preserves_chinese(self):
        assert sanitize_input("分析苹果公司的营收") == "分析苹果公司的营收"

    def test_truncates_long_input(self):
        long_input = "a" * 600
        assert len(sanitize_input(long_input)) == 500

    def test_removes_control_chars(self):
        assert sanitize_input("hello\x00world\x1f") == "helloworld"


# ─── Eval: Number Extraction ───


class TestNumberExtraction:
    def test_basic_number(self):
        nums = _extract_numbers("Revenue was 130,497 million")
        assert 130497.0 in nums

    def test_percentage(self):
        nums = _extract_numbers("Gross margin was 77.6%")
        assert 77.6 in nums

    def test_negative(self):
        nums = _extract_numbers("Growth was (976) million")
        assert -976.0 in nums

    def test_dollar_sign(self):
        nums = _extract_numbers("$8,462M debt")
        assert 8462.0 in nums

    def test_multiple_numbers(self):
        nums = _extract_numbers("From $15,005M to $115,199M, approximately 668% growth")
        assert len(nums) >= 3

    def test_no_numbers(self):
        nums = _extract_numbers("Data Center segment")
        assert len(nums) == 0


# ─── Eval: Approximate Matching ───


class TestApproxMatch:
    def test_exact(self):
        assert _approx_match(67.0, 67.0)

    def test_within_tolerance(self):
        assert _approx_match(67.3, 67.0)  # 0.45% diff

    def test_percentage_close(self):
        assert _approx_match(88.25, 88.3)

    def test_small_decimal(self):
        assert _approx_match(0.11, 0.107)  # 2.8% diff

    def test_outside_tolerance(self):
        assert not _approx_match(100.0, 90.0)  # 11% diff

    def test_zero(self):
        assert _approx_match(0.005, 0)


# ─── Eval: check_answer ───


class TestCheckAnswer:
    def test_exact_match(self):
        r = check_answer("130,497", "130,497")
        assert r["correct"]
        assert r["partial_score"] >= 0.9

    def test_approx_number(self):
        r = check_answer("67.3%", "Approximately 67% CAGR")
        assert r["correct"]

    def test_wrong_number(self):
        r = check_answer("45,274", "17,107")
        assert not r["correct"]

    def test_entity_only(self):
        r = check_answer("Data Center", "Data Center with $115,199 million")
        assert r["correct"]

    def test_hallucination(self):
        r = check_answer("Based on my analysis the revenue was approximately 500 billion dollars in the fiscal year", "130,497")
        assert r["hallucination"]

    def test_growth_synonyms(self):
        r = check_answer("NVIDIA Data Center revenue grew by 667.8%", "From $15,005M to $115,199M, approximately 668% growth")
        assert r["correct"]

    def test_debt_equity(self):
        r = check_answer("NVIDIA debt-to-equity ratio was approximately 0.1067", "Approximately 0.107 ($8,462M debt / $79,339M equity)")
        assert r["correct"]


# ─── Plot Tool (subprocess-isolated to avoid matplotlib C-level abort in pytest) ───


class TestPlotTool:
    """Plot tests run in subprocess to avoid matplotlib/pytest interaction crash."""

    @staticmethod
    def _run_plot_test(data: str, plot_type: str, title: str) -> str:
        script = f'''
import os, sys
os.chdir(r"{settings.BASE_DIR}")
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
from src.tools.plot_tool import create_plot
result = create_plot.invoke({{
    "data_str": {repr(data)},
    "plot_type": "{plot_type}",
    "title": "{title}",
    "xlabel": "X", "ylabel": "Y",
}})
print(result)
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr

    def test_bar_chart(self):
        out = self._run_plot_test('[{"Year": 2023, "Revenue": 100}, {"Year": 2024, "Revenue": 150}]', "bar", "Test Revenue")
        assert "Chart generated" in out

    def test_line_chart(self):
        out = self._run_plot_test('[{"Quarter": "Q1", "Profit": 50}, {"Quarter": "Q2", "Profit": 80}]', "line", "Profit Trend")
        assert "Chart generated" in out

    def test_pie_chart(self):
        out = self._run_plot_test('[{"Segment": "DC", "Revenue": 115199}, {"Segment": "Gaming", "Revenue": 11359}]', "pie", "Revenue by Segment")
        assert "Chart generated" in out

    def test_grouped_bar(self):
        out = self._run_plot_test(
            '[{"Company": "NVIDIA", "FY2024": 60922, "FY2025": 130497}, {"Company": "Apple", "FY2024": 391035, "FY2025": 420500}]',
            "bar",
            "Revenue Comparison",
        )
        assert "Chart generated" in out

    def test_stacked_bar(self):
        out = self._run_plot_test(
            '[{"Year": "FY2023", "DataCenter": 15005, "Gaming": 9067}, {"Year": "FY2025", "DataCenter": 115199, "Gaming": 11359}]',
            "stacked_bar",
            "Segment Revenue",
        )
        assert "Chart generated" in out


# ─── Query Rewriting ───


class TestQueryRewrite:
    def test_rewrite_returns_list(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="NVIDIA gross margin FY2025\nApple gross margin FY2025")
        result = _rewrite_query(mock_llm, "Compare NVIDIA and Apple gross margin")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_rewrite_max_3(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Q1\nQ2\nQ3\nQ4\nQ5")
        result = _rewrite_query(mock_llm, "Complex question")
        assert len(result) <= 3

    def test_rewrite_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")
        result = _rewrite_query(mock_llm, "What is revenue?")
        assert result == ["What is revenue?"]

    def test_rewrite_strips_think_tags(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="<think>Let me think...</think>\nNVIDIA revenue FY2025\nApple revenue FY2025")
        result = _rewrite_query(mock_llm, "Compare revenue")
        assert all("<think>" not in q for q in result)


# ─── Trend & Time Range Detection ───


class TestTrendDetection:
    def test_detect_fiscal_years(self):
        from src.agents.researcher import _detect_fiscal_years

        years = _detect_fiscal_years("NVIDIA revenue in FY2024 and FY2025")
        assert "2024" in years
        assert "2025" in years

    def test_detect_year_range(self):
        from src.agents.researcher import _detect_year_range

        result = _detect_year_range("NVIDIA revenue from FY2021 to FY2025")
        assert result == (2021, 2025)

    def test_detect_year_range_dash(self):
        from src.agents.researcher import _detect_year_range

        result = _detect_year_range("Revenue trend FY2022-FY2024")
        assert result == (2022, 2024)

    def test_detect_year_range_none(self):
        from src.agents.researcher import _detect_year_range

        result = _detect_year_range("What was NVIDIA revenue in FY2025?")
        assert result is None

    def test_is_trend_question(self):
        from src.agents.researcher import _is_trend_question

        assert _is_trend_question("What is the revenue growth trend?")
        assert _is_trend_question("Calculate CAGR from FY2021 to FY2025")
        assert _is_trend_question("How has revenue changed over time?")
        assert not _is_trend_question("What was NVIDIA revenue in FY2025?")

    def test_build_trend_queries(self):
        from src.agents.researcher import _build_trend_queries

        queries = _build_trend_queries("NVIDIA revenue trend from FY2023 to FY2025")
        # Should have one query per year + original fallback
        assert len(queries) >= 3
        assert any("FY2023" in q for q in queries)
        assert any("FY2025" in q for q in queries)

    def test_build_trend_queries_multi_company(self):
        from src.agents.researcher import _build_trend_queries

        queries = _build_trend_queries("Compare NVIDIA and Apple revenue growth from FY2023 to FY2025")
        # 2 companies × 3 years + 1 fallback = 7
        assert len(queries) >= 6

    def test_build_trend_queries_no_range(self):
        from src.agents.researcher import _build_trend_queries

        queries = _build_trend_queries("What was NVIDIA revenue?")
        assert queries == ["What was NVIDIA revenue?"]


# ─── Concurrent Query Execution ───


class TestConcurrentQuery:
    def test_concurrent_multi_query(self):
        """Multi-entity queries should use concurrent execution."""
        import time
        from unittest.mock import MagicMock, patch

        mock_llm = MagicMock()
        mock_adapter = MagicMock()

        # Simulate 0.1s per query
        def slow_query(q):
            time.sleep(0.1)
            return {"model_answer": f"Answer for {q}", "source_nodes": ["src"]}

        mock_adapter.query.side_effect = slow_query

        from src.agents.researcher import create_researcher_node

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "Compare NVIDIA and Apple revenue in FY2025")]})

        # Should have called query multiple times
        assert mock_adapter.query.call_count >= 2
        assert result["sender"] == "Researcher"

    def test_single_query_no_concurrency(self):
        """Single queries should not use ThreadPoolExecutor overhead."""
        from unittest.mock import MagicMock, patch

        mock_llm = MagicMock()
        mock_adapter = MagicMock()
        mock_adapter.query.return_value = {"model_answer": "Revenue was $130B", "source_nodes": ["src"]}

        from src.agents.researcher import create_researcher_node

        with patch("src.rag_adapter.adapter", mock_adapter):
            node = create_researcher_node(mock_llm)
            result = node({"messages": [("user", "What was NVIDIA revenue in FY2025?")]})

        assert mock_adapter.query.call_count >= 1
        assert result["sender"] == "Researcher"
