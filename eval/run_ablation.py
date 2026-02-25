"""
Ablation study for LangGraph Financial Swarm.
Tests the contribution of each component by disabling them individually.

Ablation modes:
  full          - Full system (baseline for comparison)
  no_rewrite    - Disable query rewriting (always use original question)
  no_structured - Disable structure-aware chunking (use SentenceSplitter)
  no_multiquery - Disable multi-query synthesis (single query even for complex Qs)
  all           - Run all ablation modes

Usage:
    python eval/run_ablation.py --mode all
    python eval/run_ablation.py --mode no_rewrite
    python eval/run_ablation.py --mode full --limit 5
"""

import argparse
import asyncio
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.core.config import settings
from src.core.llm import DirectOllamaChat

console = Console()

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results" / "ablation"


def load_benchmark():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Answer checking (same as run_eval.py) ──────────────────────────


def _extract_numbers(text: str) -> list[float]:
    clean = text.replace(",", "").replace("$", "").replace("(", "-").replace(")", "")
    raw = re.findall(r"-?[\d]+\.?\d*%?", clean)
    nums = []
    for r in raw:
        try:
            if r.endswith("%"):
                nums.append(float(r[:-1]))
            else:
                nums.append(float(r))
        except ValueError:
            continue
    return nums


def _approx_match(a: float, b: float, tol: float = 0.05) -> bool:
    if a == b:
        return True
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / max(abs(b), 1e-9) <= tol


def _extract_entities(text: str) -> set[str]:
    entities = set()
    lower = text.lower()
    for name in [
        "data center",
        "gaming",
        "automotive",
        "professional visualization",
        "united states",
        "china",
        "taiwan",
        "singapore",
        "nvidia",
        "apple",
        "microsoft",
        "intelligent cloud",
        "iphone",
        "mac",
        "services",
    ]:
        if name in lower:
            entities.add(name)
    for m in re.finditer(r"fy\d{4}", lower):
        entities.add(m.group())
    growth_terms = ["increase", "increased", "grew", "growth", "grow", "up from", "higher"]
    decline_terms = ["decrease", "decreased", "decline", "declined", "dropped", "drop", "lower", "fell"]
    compare_terms = ["better", "worse", "fluctuat", "not consistently", "inconsistent"]
    if any(t in lower for t in growth_terms):
        entities.add("growth")
    if any(t in lower for t in decline_terms):
        entities.add("decline")
    if any(t in lower for t in compare_terms):
        entities.add("comparison")
    return entities


def check_answer(predicted: str, ground_truth: str) -> dict:
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()
    gt_nums = _extract_numbers(gt_lower)
    pred_nums = _extract_numbers(pred_lower)
    matched = 0
    for gn in gt_nums:
        for pn in pred_nums:
            if _approx_match(pn, gn):
                matched += 1
                break
    total_nums = max(len(gt_nums), 1)
    if matched == len(gt_nums):
        number_score = 1.0
    elif matched > 0:
        number_score = max(0.5, matched / total_nums)
    else:
        number_score = 0.0
    gt_entities = _extract_entities(gt_lower)
    pred_entities = _extract_entities(pred_lower)
    if gt_entities:
        entity_overlap = len(gt_entities & pred_entities) / len(gt_entities)
    else:
        entity_overlap = 1.0 if number_score > 0 else 0.0
    if gt_nums and gt_entities:
        partial_score = 0.5 * number_score + 0.5 * entity_overlap
    elif gt_nums:
        partial_score = 0.8 * number_score + 0.2 * entity_overlap
    else:
        partial_score = 0.3 * number_score + 0.7 * entity_overlap
    if entity_overlap >= 0.8 and number_score >= 0.5:
        partial_score = max(partial_score, 0.75)
    if entity_overlap >= 0.8 and len(gt_nums) <= 1:
        partial_score = max(partial_score, 0.6)
    correct = partial_score >= 0.45
    is_hallucination = number_score == 0 and entity_overlap < 0.3 and len(pred_lower) > 50 and "error" not in pred_lower
    reasons = []
    if number_score == 1.0:
        reasons.append(f"All {len(gt_nums)} numbers matched")
    elif number_score > 0:
        reasons.append(f"{matched}/{len(gt_nums)} numbers matched (approx)")
    else:
        reasons.append("No numbers matched")
    if entity_overlap > 0:
        reasons.append(f"Entity overlap: {entity_overlap:.0%}")
    return {
        "correct": correct,
        "hallucination": is_hallucination,
        "partial_score": round(partial_score, 3),
        "number_recall": round(number_score, 3),
        "entity_overlap": round(entity_overlap, 3),
        "qualitative_match": entity_overlap > 0.3,
        "reasoning": "; ".join(reasons),
    }


# ── Ablation RAG Adapter (naive chunking) ──────────────────────────


class NaiveRAGAdapter:
    """RAG with SentenceSplitter instead of structure-aware chunking."""

    def __init__(self):
        self.query_engine = None

    def _build_index(self):
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        from src.core.llm_index import DirectOllamaLLM
        from src.utils.robustness import log_agent_action

        log_agent_action("NaiveRAG", "Init", "Building naive (SentenceSplitter) index...")

        embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        llm = DirectOllamaLLM(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            request_timeout=settings.LLM_TIMEOUT,
        )

        data_path = settings.RAG_DATA_PATH
        with open(data_path, encoding="utf-8") as f:
            raw_text = f.read()

        # Naive: just split by sentences, no structure awareness
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        documents = [Document(text=raw_text)]

        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
            transformations=[splitter],
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=settings.RAG_TOP_K,
            llm=llm,
        )

    def query(self, question: str) -> dict[str, Any]:
        if self.query_engine is None:
            self._build_index()
        response = self.query_engine.query(question)
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                content = node.node.get_content()[:150]
                sources.append(content)
        return {
            "model_answer": str(response),
            "source_nodes": sources,
        }


# ── Ablation Researcher Variants ───────────────────────────────────


def create_researcher_no_rewrite(llm) -> Callable:
    """Researcher without query rewriting — always uses original question."""
    from langchain_core.messages import AIMessage

    from src.utils.robustness import log_agent_action

    def researcher_node(state) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break
        if not user_question:
            return {"messages": [AIMessage(content="No question found.")], "sender": "Researcher"}

        from src.rag_adapter import adapter

        log_agent_action("Researcher[no_rewrite]", "RAG Query", f"Q: {user_question}")
        try:
            result = adapter.query(user_question)
            answer = result.get("model_answer", "No answer found.")
        except Exception as e:
            answer = f"Error: {e}"
        # Ensure answer is long enough for Supervisor FINISH detection
        if len(answer) < 30:
            answer = f"Answer to '{user_question}': {answer}"
        log_agent_action("Researcher[no_rewrite]", "Answer", answer[:200])
        return {"messages": [AIMessage(content=answer)], "sender": "Researcher"}

    return researcher_node


def create_researcher_no_multiquery(llm) -> Callable:
    """Researcher with rewriting but no multi-query synthesis — uses only first sub-query."""
    from langchain_core.messages import AIMessage

    from src.agents.researcher import _rewrite_query
    from src.utils.robustness import log_agent_action

    def researcher_node(state) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break
        if not user_question:
            return {"messages": [AIMessage(content="No question found.")], "sender": "Researcher"}

        from src.rag_adapter import adapter

        # Still rewrite, but only use the FIRST sub-query (no multi-query merge)
        complex_indicators = ["compare", "vs", "versus", "and", "which company", "highest", "lowest", "between", "all three"]
        needs_rewrite = any(ind in user_question.lower() for ind in complex_indicators)

        if needs_rewrite:
            sub_queries = _rewrite_query(llm, user_question)
            query_to_use = sub_queries[0] if sub_queries else user_question
            log_agent_action("Researcher[no_multi]", "Single Query", f"Using: {query_to_use}")
        else:
            query_to_use = user_question

        try:
            result = adapter.query(query_to_use)
            answer = result.get("model_answer", "No answer found.")
        except Exception as e:
            answer = f"Error: {e}"
        if len(answer) < 30:
            answer = f"Answer to '{user_question}': {answer}"
        log_agent_action("Researcher[no_multi]", "Answer", answer[:200])
        return {"messages": [AIMessage(content=answer)], "sender": "Researcher"}

    return researcher_node


def create_researcher_no_structured(llm) -> Callable:
    """Researcher using naive RAG (no structured chunking), with full rewrite+multiquery."""
    from langchain_core.messages import AIMessage, SystemMessage

    from src.agents.researcher import _rewrite_query
    from src.utils.robustness import log_agent_action

    naive_adapter = NaiveRAGAdapter()

    def researcher_node(state) -> dict[str, Any]:
        messages = state.get("messages", [])
        user_question = ""
        for msg in messages:
            if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "user":
                user_question = msg[1]
                break
            if hasattr(msg, "type") and msg.type == "human":
                user_question = msg.content
                break
        if not user_question:
            return {"messages": [AIMessage(content="No question found.")], "sender": "Researcher"}

        complex_indicators = ["compare", "vs", "versus", "and", "which company", "highest", "lowest", "between", "all three"]
        needs_rewrite = any(ind in user_question.lower() for ind in complex_indicators)

        if needs_rewrite:
            sub_queries = _rewrite_query(llm, user_question)
            log_agent_action("Researcher[no_struct]", "Rewrite", f"{len(sub_queries)} sub-queries")
        else:
            sub_queries = [user_question]

        all_answers = []
        for sq in sub_queries:
            log_agent_action("Researcher[no_struct]", "RAG Query", f"Q: {sq}")
            try:
                result = naive_adapter.query(sq)
                all_answers.append(result.get("model_answer", "No answer."))
            except Exception as e:
                all_answers.append(f"Error: {e}")

        if len(sub_queries) > 1 and len(all_answers) > 1:
            combined = "\n\n".join(f"[Sub-query: {sq}]\n{ans}" for sq, ans in zip(sub_queries, all_answers, strict=False))
            try:
                prompt = (
                    f"Based on the following retrieved data, answer the original question.\n\n"
                    f"Original question: {user_question}\n\n"
                    f"Retrieved data:\n{combined}\n\n"
                    f"Provide a clear, concise answer with specific numbers."
                )
                response = llm.invoke([SystemMessage(content=prompt)])
                final = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL).strip()
            except Exception:
                final = "\n\n".join(all_answers)
        else:
            final = all_answers[0] if all_answers else "No answer."

        log_agent_action("Researcher[no_struct]", "Answer", final[:200])
        if len(final) < 30:
            final = f"Answer to '{user_question}': {final}"
        return {"messages": [AIMessage(content=final)], "sender": "Researcher"}

    return researcher_node


# ── Graph Builder ───────────────────────────────────────────────────


def build_graph(llm, researcher_factory: Callable):
    """Build the LangGraph workflow with a given researcher variant."""
    from langgraph.graph import END, START, StateGraph

    from src.agents.chart_gen import create_quant_node
    from src.agents.supervisor import create_supervisor_node
    from src.core.types import AgentState

    members = ["Researcher", "Quant"]
    supervisor_node = create_supervisor_node(llm, members)
    researcher_node = researcher_factory(llm)
    quant_node = create_quant_node(llm)

    workflow = StateGraph(AgentState)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Quant", quant_node)

    workflow.add_conditional_edges(
        "Supervisor",
        lambda x: x["next"],
        {"Researcher": "Researcher", "Quant": "Quant", "FINISH": END},
    )
    workflow.add_edge("Researcher", "Supervisor")
    workflow.add_edge("Quant", "Supervisor")
    workflow.add_edge(START, "Supervisor")

    return workflow.compile()


# ── Run One Ablation ────────────────────────────────────────────────


async def run_ablation(llm, questions: list, mode: str) -> list:
    """Run evaluation for a specific ablation mode."""
    # Clear RAG cache to avoid cross-contamination
    import diskcache as dc

    settings.DATA_DIR / "rag_cache"
    if mode != "full":
        # Use separate cache per ablation mode
        dc.Cache(settings.DATA_DIR / f"rag_cache_ablation_{mode}")

    # Select researcher variant
    if mode == "full":
        from src.agents.researcher import create_researcher_node

        researcher_factory = create_researcher_node
    elif mode == "no_rewrite":
        researcher_factory = create_researcher_no_rewrite
    elif mode == "no_multiquery":
        researcher_factory = create_researcher_no_multiquery
    elif mode == "no_structured":
        researcher_factory = create_researcher_no_structured
    else:
        raise ValueError(f"Unknown ablation mode: {mode}")

    graph = build_graph(llm, researcher_factory)

    results = []
    label = f"[cyan]Ablation: {mode}[/]"
    for q in track(questions, description=label):
        start = time.time()
        try:
            final_state = await graph.ainvoke(
                {"messages": [("user", q["question"])]},
                {"recursion_limit": 25},
            )
            answer = final_state["messages"][-1].content if final_state["messages"] else "No answer"
        except Exception as e:
            answer = f"Error: {e}"
        latency = time.time() - start

        check = check_answer(answer, q["answer"])
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["answer"],
                "predicted": answer[:500],
                "difficulty": q.get("difficulty", "unknown"),
                "type": q.get("type", "unknown"),
                "latency_s": round(latency, 2),
                **check,
            }
        )
    return results


# ── Summary & Comparison ───────────────────────────────────────────


def print_summary(name: str, results: list):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    hallucinated = sum(1 for r in results if r["hallucination"])
    avg_latency = sum(r["latency_s"] for r in results) / max(total, 1)
    avg_partial = sum(r.get("partial_score", 0) for r in results) / max(total, 1)

    table = Table(title=f"{name} ({total} questions)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{correct}/{total} ({100 * correct / total:.1f}%)")
    table.add_row("Avg Partial Score", f"{avg_partial:.3f}")
    table.add_row("Hallucination Rate", f"{hallucinated}/{total} ({100 * hallucinated / total:.1f}%)")
    table.add_row("Avg Latency", f"{avg_latency:.2f}s")

    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            ps = sum(r.get("partial_score", 0) for r in subset) / len(subset)
            table.add_row(f"  {diff.capitalize()}", f"{c}/{len(subset)} (ps={ps:.2f})")

    for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
        subset = [r for r in results if r.get("type") == qtype]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            ps = sum(r.get("partial_score", 0) for r in subset) / len(subset)
            table.add_row(f"  {qtype}", f"{c}/{len(subset)} (ps={ps:.2f})")

    console.print(table)


def print_comparison(all_results: dict[str, list]):
    """Print side-by-side comparison of all ablation modes."""
    table = Table(title="Ablation Study Comparison")
    table.add_column("Mode", style="bold")
    table.add_column("Accuracy", justify="right")
    table.add_column("Partial Score", justify="right")
    table.add_column("Hallucination", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Easy", justify="right")
    table.add_column("Medium", justify="right")
    table.add_column("Hard", justify="right")

    for mode, results in all_results.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        avg_ps = sum(r.get("partial_score", 0) for r in results) / max(total, 1)
        hall = sum(1 for r in results if r["hallucination"])
        avg_lat = sum(r["latency_s"] for r in results) / max(total, 1)

        easy = [r for r in results if r.get("difficulty") == "easy"]
        med = [r for r in results if r.get("difficulty") == "medium"]
        hard = [r for r in results if r.get("difficulty") == "hard"]

        e_acc = f"{sum(1 for r in easy if r['correct'])}/{len(easy)}" if easy else "-"
        m_acc = f"{sum(1 for r in med if r['correct'])}/{len(med)}" if med else "-"
        h_acc = f"{sum(1 for r in hard if r['correct'])}/{len(hard)}" if hard else "-"

        table.add_row(
            mode,
            f"{correct}/{total} ({100 * correct / total:.1f}%)",
            f"{avg_ps:.3f}",
            f"{hall}/{total} ({100 * hall / total:.1f}%)",
            f"{avg_lat:.1f}s",
            e_acc,
            m_acc,
            h_acc,
        )

    console.print(table)


def save_results(mode: str, results: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{mode}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Results saved to {path}[/]")


def save_comparison(all_results: dict[str, list]):
    """Save comparison summary as JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for mode, results in all_results.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        avg_ps = sum(r.get("partial_score", 0) for r in results) / max(total, 1)
        hall = sum(1 for r in results if r["hallucination"])
        avg_lat = sum(r["latency_s"] for r in results) / max(total, 1)

        by_diff = {}
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in results if r.get("difficulty") == diff]
            if subset:
                by_diff[diff] = {
                    "correct": sum(1 for r in subset if r["correct"]),
                    "total": len(subset),
                    "avg_partial": round(sum(r.get("partial_score", 0) for r in subset) / len(subset), 3),
                }

        by_type = {}
        for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
            subset = [r for r in results if r.get("type") == qtype]
            if subset:
                by_type[qtype] = {
                    "correct": sum(1 for r in subset if r["correct"]),
                    "total": len(subset),
                    "avg_partial": round(sum(r.get("partial_score", 0) for r in subset) / len(subset), 3),
                }

        summary[mode] = {
            "accuracy": f"{correct}/{total}",
            "accuracy_pct": round(100 * correct / total, 1),
            "avg_partial_score": round(avg_ps, 3),
            "hallucination_rate": round(100 * hall / total, 1),
            "avg_latency_s": round(avg_lat, 1),
            "by_difficulty": by_diff,
            "by_type": by_type,
        }

    path = RESULTS_DIR / "comparison.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Comparison saved to {path}[/]")


# ── Main ────────────────────────────────────────────────────────────

ABLATION_MODES = ["full", "no_rewrite", "no_structured", "no_multiquery"]


async def main():
    parser = argparse.ArgumentParser(description="Ablation Study for Financial Swarm")
    parser.add_argument("--mode", choices=ABLATION_MODES + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit questions (0=all)")
    args = parser.parse_args()

    questions = load_benchmark()
    if args.limit > 0:
        questions = questions[: args.limit]

    modes = ABLATION_MODES if args.mode == "all" else [args.mode]

    console.print(f"\n[bold]Ablation Study: {len(questions)} questions, modes: {modes}[/]\n")

    llm = DirectOllamaChat(
        model=settings.LLM_MODEL,
        temperature=0.0,
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )

    all_results = {}
    for mode in modes:
        console.print(f"\n[bold yellow]── Running: {mode} ──[/]\n")
        results = await run_ablation(llm, questions, mode)
        print_summary(f"Ablation: {mode}", results)
        save_results(mode, results)
        all_results[mode] = results

    if len(all_results) > 1:
        console.print("\n[bold yellow]── Comparison ──[/]\n")
        print_comparison(all_results)
        save_comparison(all_results)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
