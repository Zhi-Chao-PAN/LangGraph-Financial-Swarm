"""
Rich CLI display components for query results, tables, and exports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.ui import console


def format_answer_panel(question: str, answer: str, confidence: float | None = None, latency: float | None = None) -> Panel:
    """Create a Rich Panel for a query answer."""
    # Split answer and citations
    parts = answer.split("**Citations**:")
    main_answer = parts[0].strip()
    citations = parts[1].strip() if len(parts) > 1 else ""

    # Build content
    content = Text()
    content.append(main_answer, style="white")

    if citations:
        content.append("\n\n")
        content.append("📚 Citations:\n", style="bold cyan")
        for line in citations.strip().split("\n"):
            content.append(f"  {line}\n", style="dim")

    # Build subtitle
    subtitle_parts = []
    if confidence is not None:
        color = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.6 else "red"
        subtitle_parts.append(f"[{color}]Confidence: {confidence:.0%}[/]")
    if latency is not None:
        subtitle_parts.append(f"[dim]Latency: {latency:.1f}s[/]")

    subtitle = " │ ".join(subtitle_parts) if subtitle_parts else None

    return Panel(
        content,
        title=f"[bold]Q: {question[:80]}{'...' if len(question) > 80 else ''}[/]",
        subtitle=subtitle,
        border_style="green",
        padding=(1, 2),
    )


def print_answer(question: str, answer: str, confidence: float | None = None, latency: float | None = None) -> None:
    """Print a formatted answer panel."""
    panel = format_answer_panel(question, answer, confidence, latency)
    console.print(panel)
    console.print()


def print_comparison_table(data: list[dict[str, Any]], title: str = "Comparison") -> None:
    """Print a comparison table from structured data."""
    if not data:
        console.print("[warning]No data to display.[/]")
        return

    table = Table(title=title, show_header=True, header_style="bold cyan", border_style="blue")

    # Add columns from first row keys
    for key in data[0]:
        table.add_column(key, style="bold" if key in ("Company", "Metric") else "")

    for row in data:
        table.add_row(*[str(v) for v in row.values()])

    console.print(table)
    console.print()


def print_eval_results(results: list[dict[str, Any]], title: str = "Evaluation Results") -> None:
    """Print evaluation results in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold cyan", border_style="blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Question", width=50)
    table.add_column("Correct", width=10, justify="center")
    table.add_column("Score", width=10, justify="center")
    table.add_column("Halluc.", width=10, justify="center")

    for i, r in enumerate(results, 1):
        correct = "[success]✓[/]" if r.get("correct") else "[error]✗[/]"
        score = f"{r.get('partial_score', 0):.2f}"
        halluc = "[error]Yes[/]" if r.get("hallucination") else "[success]No[/]"
        q = r.get("question", "")[:48]
        table.add_row(str(i), q, correct, score, halluc)

    console.print(table)

    # Summary row
    total = len(results)
    correct_count = sum(1 for r in results if r.get("correct"))
    avg_score = sum(r.get("partial_score", 0) for r in results) / max(total, 1)
    halluc_count = sum(1 for r in results if r.get("hallucination"))

    summary = Table(show_header=False, border_style="green", padding=(0, 2))
    summary.add_column("", width=20)
    summary.add_column("", width=20)
    summary.add_column("", width=20)
    summary.add_column("", width=20)
    summary.add_row(
        f"[bold]Total: {total}[/]",
        f"[success]Correct: {correct_count}/{total} ({correct_count / max(total, 1):.0%})[/]",
        f"[metric]Avg Score: {avg_score:.3f}[/]",
        f"[{'error' if halluc_count else 'success'}]Hallucinations: {halluc_count}[/]",
    )
    console.print(summary)
    console.print()


def export_results(results: list[dict[str, Any]], output_path: str, fmt: str = "json") -> None:
    """Export query/eval results to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif fmt == "markdown":
        lines = [f"# Results Export ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"## Q{i}: {r.get('question', 'N/A')}\n")
            lines.append(f"**Answer:** {r.get('answer', 'N/A')}\n")
            if "correct" in r:
                lines.append(f"**Correct:** {'Yes' if r['correct'] else 'No'} | **Score:** {r.get('partial_score', 0):.2f}\n")
            lines.append("---\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    console.print(f"[success]Exported {len(results)} results to {path}[/]")


def print_cache_stats(cache_dir: Path) -> None:
    """Print cache statistics."""
    if not cache_dir.exists():
        console.print("[warning]Cache directory not found.[/]")
        return

    files = list(cache_dir.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    table = Table(title="Cache Statistics", show_header=True, header_style="bold cyan", border_style="blue")
    table.add_column("Metric", style="bold", width=20)
    table.add_column("Value", width=30)
    table.add_row("Entries", str(len(files)))
    table.add_row("Total Size", f"{total_size / 1024:.1f} KB")
    table.add_row("Location", str(cache_dir))

    console.print(table)
    console.print()


def print_query_stats(stats: dict[str, Any]) -> None:
    """Print query performance statistics."""
    table = Table(title="Query Statistics", show_header=True, header_style="bold cyan", border_style="blue")
    table.add_column("Metric", style="bold", width=25)
    table.add_column("Value", width=20)

    table.add_row("Total Queries", str(stats.get("total_queries", 0)))
    table.add_row("Avg Latency", f"{stats.get('avg_latency', 0):.2f}s")
    table.add_row("Cache Hit Rate", f"{stats.get('cache_hit_rate', 0):.0%}")
    table.add_row("Avg Confidence", f"{stats.get('avg_confidence', 0):.2f}")

    console.print(table)
    console.print()
