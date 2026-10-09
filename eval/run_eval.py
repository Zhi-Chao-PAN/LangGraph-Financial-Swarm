"""
Evaluation script for LangGraph Financial Swarm.
Compares Swarm (multi-agent RAG) vs Baseline (direct LLM) on financial QA.

Usage:
    python eval/run_eval.py --mode swarm
    python eval/run_eval.py --mode baseline
    python eval/run_eval.py --mode both
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track
from rich.table import Table

from eval.eval_utils import check_answer, compute_mrr, compute_ndcg
from src.core.config import settings
from src.core.llm import DirectOllamaChat

console = Console()

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_benchmark():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)





async def run_baseline(llm, questions: list) -> list:
    """Direct LLM QA without RAG."""
    results = []
    for q in track(questions, description="[blue]Baseline (Direct LLM)..."):
        start = time.time()
        try:
            response = llm.invoke(f"Answer this financial question concisely: {q['question']}")
            answer = response.content
        except Exception as e:
            answer = f"Error: {e}"
        latency = time.time() - start

        check = check_answer(answer, q["answer"])
        results.append(
            {
                "id": q["id"],
                "question": q["question"],
                "ground_truth": q["answer"],
                "predicted": answer,
                "difficulty": q.get("difficulty", "unknown"),
                "type": q.get("type", "unknown"),
                "latency_s": round(latency, 2),
                **check,
            }
        )
    return results


async def run_swarm(llm, questions: list) -> list:
    """Multi-agent Swarm with RAG."""
    from langgraph.graph import END, START, StateGraph

    from src.agents.chart_gen import create_quant_node
    from src.agents.researcher import create_researcher_node
    from src.agents.supervisor import create_supervisor_node
    from src.core.types import AgentState

    members = ["Researcher", "Quant"]
    supervisor_node = create_supervisor_node(llm, members)
    researcher_node = create_researcher_node(llm)
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
    graph = workflow.compile()

    results = []
    for q in track(questions, description="[green]Swarm (Multi-Agent RAG)..."):
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

    # IR Metrics: nDCG@5 and MRR (Improvement 7)
    relevance_lists = []
    for r in results:
        relevance = [1.0 if r["correct"] else r.get("partial_score", 0.0)]
        relevance_lists.append(relevance)

    ndcg = sum(compute_ndcg(rl, k=5) for rl in relevance_lists) / max(total, 1)
    mrr = compute_mrr(relevance_lists)
    table.add_row("nDCG@5", f"{ndcg:.3f}")
    table.add_row("MRR", f"{mrr:.3f}")

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r.get("difficulty") == diff]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            ps = sum(r.get("partial_score", 0) for r in subset) / len(subset)
            table.add_row(f"  {diff.capitalize()}", f"{c}/{len(subset)} (ps={ps:.2f})")

    # By type
    for qtype in ["single_hop", "comparison", "calculation", "multi_hop"]:
        subset = [r for r in results if r.get("type") == qtype]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            ps = sum(r.get("partial_score", 0) for r in subset) / len(subset)
            table.add_row(f"  {qtype}", f"{c}/{len(subset)} (ps={ps:.2f})")

    console.print(table)


def save_results(name: str, results: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name.lower().replace(' ', '_')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"[dim]Results saved to {path}[/]")


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Financial Swarm")
    parser.add_argument("--mode", choices=["swarm", "baseline", "both"], default="both")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0=all)")
    args = parser.parse_args()

    questions = load_benchmark()
    if args.limit > 0:
        questions = questions[: args.limit]

    console.print(f"\n[bold]Evaluating {len(questions)} questions[/]\n")

    llm = DirectOllamaChat(
        model=settings.LLM_MODEL,
        temperature=0.0,
        base_url=settings.LLM_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )

    if args.mode in ("baseline", "both"):
        baseline_results = await run_baseline(llm, questions)
        print_summary("Baseline (Direct LLM)", baseline_results)
        save_results("baseline", baseline_results)

    if args.mode in ("swarm", "both"):
        swarm_results = await run_swarm(llm, questions)
        print_summary("Swarm (Multi-Agent RAG)", swarm_results)
        save_results("swarm", swarm_results)

    if args.mode == "both":
        console.print("\n[bold yellow]Comparison[/]")
        b_acc = sum(1 for r in baseline_results if r["correct"]) / len(baseline_results) * 100
        s_acc = sum(1 for r in swarm_results if r["correct"]) / len(swarm_results) * 100
        b_ps = sum(r.get("partial_score", 0) for r in baseline_results) / len(baseline_results)
        s_ps = sum(r.get("partial_score", 0) for r in swarm_results) / len(swarm_results)
        console.print(f"  Baseline: {b_acc:.1f}% acc, {b_ps:.3f} partial")
        console.print(f"  Swarm:    {s_acc:.1f}% acc, {s_ps:.3f} partial")
        console.print(f"  Delta:    {s_acc - b_acc:+.1f}% acc")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
