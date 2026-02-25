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
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.core.config import settings
from src.core.llm import DirectOllamaChat

console = Console()

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_benchmark():
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from text, normalizing commas and percentages."""
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
    """Check if two numbers are approximately equal within 5% tolerance."""
    if a == b:
        return True
    if b == 0:
        return abs(a) < 0.01
    return abs(a - b) / max(abs(b), 1e-9) <= tol


def _extract_entities(text: str) -> set[str]:
    """Extract key financial entities from text."""
    entities = set()
    lower = text.lower()

    # Segment/region names
    for name in ["data center", "gaming", "automotive", "professional visualization", "united states", "china", "taiwan", "singapore"]:
        if name in lower:
            entities.add(name)

    # Fiscal years
    for m in re.finditer(r"fy\d{4}", lower):
        entities.add(m.group())

    # Directional terms (group synonyms)
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
    """
    Robust answer checking with approximate number matching and entity matching.
    Returns partial_score (0-1), correct (bool), and detailed reasoning.
    """
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()

    gt_nums = _extract_numbers(gt_lower)
    pred_nums = _extract_numbers(pred_lower)

    # Number matching with tolerance
    # Score based on: did we find at least the KEY numbers?
    # If prediction matches ANY gt number, that's a good sign.
    matched = 0
    for gn in gt_nums:
        for pn in pred_nums:
            if _approx_match(pn, gn):
                matched += 1
                break

    total_nums = max(len(gt_nums), 1)
    # Use lenient scoring: if we matched at least 1 key number, give credit
    if matched == len(gt_nums):
        number_score = 1.0
    elif matched > 0:
        # Partial credit: at least some numbers matched
        number_score = max(0.5, matched / total_nums)
    else:
        number_score = 0.0

    # Entity matching
    gt_entities = _extract_entities(gt_lower)
    pred_entities = _extract_entities(pred_lower)
    if gt_entities:
        entity_overlap = len(gt_entities & pred_entities) / len(gt_entities)
    else:
        entity_overlap = 1.0 if number_score > 0 else 0.0

    # Composite partial score
    if gt_nums and gt_entities:
        partial_score = 0.5 * number_score + 0.5 * entity_overlap
    elif gt_nums:
        partial_score = 0.8 * number_score + 0.2 * entity_overlap
    else:
        partial_score = 0.3 * number_score + 0.7 * entity_overlap

    # Boost: if entity overlap is perfect and at least 1 number matched, likely correct
    if entity_overlap >= 0.8 and number_score >= 0.5:
        partial_score = max(partial_score, 0.75)
    # Boost: perfect entity match with no numbers in gt
    if entity_overlap >= 0.8 and len(gt_nums) <= 1:
        partial_score = max(partial_score, 0.6)

    correct = partial_score >= 0.45
    is_hallucination = number_score == 0 and entity_overlap < 0.3 and len(pred_lower) > 50 and "error" not in pred_lower

    # Build reasoning
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
