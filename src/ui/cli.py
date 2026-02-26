"""
Interactive CLI for LangGraph Financial Swarm.
Provides a polished terminal interface with commands and query mode.

Usage:
    python -m src.ui.cli
"""

import shutil
import time

from rich.prompt import Prompt

from src.core.config import settings
from src.ui import console, print_banner, print_status_dashboard
from src.ui.display import (
    export_results,
    print_answer,
    print_cache_stats,
    print_query_stats,
)
from src.utils.robustness import check_ollama_health, log_agent_action

# Session statistics
_session_stats: dict = {
    "total_queries": 0,
    "total_latency": 0.0,
    "cache_hits": 0,
    "confidences": [],
    "history": [],
}

HELP_TEXT = """
[bold cyan]Available Commands:[/]
  [bold]/ask[/] <question>    Ask a financial question
  [bold]/chart[/] <question>  Request a chart/visualization
  [bold]/templates[/]         List available query templates
  [bold]/template[/] <name>   Use a query template
  [bold]/batch[/] <file>      Run batch queries from file (.txt/.json)
  [bold]/fetch[/] <TICKER>    Fetch real-time data for a company (e.g. AAPL, TSLA)
  [bold]/history[/]           Show conversation history
  [bold]/stats[/]             Show query statistics
  [bold]/cache[/]             Show cache statistics
  [bold]/cache clear[/]       Clear the RAG cache
  [bold]/export[/] <path>     Export results (json/markdown)
  [bold]/status[/]            Show system status
  [bold]/help[/]              Show this help message
  [bold]/quit[/]              Exit the application

[dim]Or just type a question directly to query the system.[/]
"""


def _handle_command(cmd: str, args: str) -> bool:
    """Handle a CLI command. Returns False to quit."""
    if cmd == "/quit" or cmd == "/exit":
        console.print("[dim]Goodbye! 👋[/]")
        return False

    if cmd == "/help":
        console.print(HELP_TEXT)

    elif cmd == "/status":
        health = check_ollama_health()
        print_status_dashboard(health, settings)

    elif cmd == "/stats":
        stats = {
            "total_queries": _session_stats["total_queries"],
            "avg_latency": _session_stats["total_latency"] / max(_session_stats["total_queries"], 1),
            "cache_hit_rate": _session_stats["cache_hits"] / max(_session_stats["total_queries"], 1),
            "avg_confidence": sum(_session_stats["confidences"]) / max(len(_session_stats["confidences"]), 1),
        }
        print_query_stats(stats)

    elif cmd == "/cache":
        if args.strip() == "clear":
            cache_dir = settings.DATA_DIR / "rag_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                console.print("[success]Cache cleared.[/]")
            else:
                console.print("[warning]No cache to clear.[/]")
        else:
            print_cache_stats(settings.DATA_DIR / "rag_cache")

    elif cmd == "/history":
        if not _session_stats["history"]:
            console.print("[dim]No queries yet this session.[/]")
        else:
            for i, h in enumerate(_session_stats["history"], 1):
                console.print(f"[bold]Q{i}:[/] {h['question']}")
                console.print(f"[dim]  → {h['answer'][:100]}...[/]")
            console.print()

    elif cmd == "/export":
        if not _session_stats["history"]:
            console.print("[warning]No results to export.[/]")
        else:
            path = args.strip() or "output/export.json"
            fmt = "markdown" if path.endswith(".md") else "json"
            export_results(_session_stats["history"], path, fmt=fmt)

    elif cmd in ("/ask", "/chart"):
        if not args.strip():
            console.print("[warning]Please provide a question.[/]")
        else:
            _execute_query(args.strip(), chart_mode=(cmd == "/chart"))

    elif cmd == "/templates":
        from rich.table import Table as RichTable

        from src.core.templates import TEMPLATE_CATEGORIES

        for cat, templates in TEMPLATE_CATEGORIES.items():
            table = RichTable(title=f"{cat.replace('_', ' ').title()}", border_style="blue")
            table.add_column("Name", style="bold")
            table.add_column("Description")
            table.add_column("Difficulty", justify="center")
            for t in templates:
                diff_style = {"easy": "green", "medium": "yellow", "hard": "red"}.get(t.difficulty, "white")
                table.add_row(t.name, t.description, f"[{diff_style}]{t.difficulty}[/]")
            console.print(table)
        console.print()

    elif cmd == "/template":
        from src.core.templates import get_template

        name = args.strip().split()[0] if args.strip() else ""
        template = get_template(name)
        if not template:
            console.print(f"[warning]Template '{name}' not found. Use /templates to list.[/]")
        else:
            console.print(f"[dim]Template: {template.template}[/]")
            console.print(f"[dim]Params: {', '.join(template.params)}[/]")
            params = {}
            for p in template.params:
                val = Prompt.ask(f"  [cyan]{p}[/]")
                params[p] = val
            rendered = template.render(**params)
            console.print(f"[bold]Query: {rendered}[/]")
            _execute_query(rendered)

    elif cmd == "/batch":
        from src.core.batch import load_queries_from_file, run_batch

        file_path = args.strip()
        if not file_path:
            console.print("[warning]Usage: /batch <file.txt|file.json>[/]")
            console.print("[dim]  File formats: .txt (one query per line, # comments) or .json (list of strings)[/]")
        else:
            try:
                queries = load_queries_from_file(file_path)
                console.print(f"[bold]Loaded {len(queries)} queries from {file_path}[/]")

                from rich.progress import Progress

                with Progress(console=console) as progress:
                    task = progress.add_task("Running batch...", total=len(queries))

                    def on_progress(i, total, q, _result):
                        progress.update(task, completed=i)

                    report = run_batch(queries, progress_callback=on_progress)

                console.print(f"\n[bold green]Batch complete:[/] {report.success}/{report.total} succeeded")
                console.print(f"[dim]Total time: {report.total_time:.1f}s | Avg: {report.avg_time:.1f}s[/]")

                # Show summary
                for r in report.results:
                    status = "[green]✓[/]" if r.success else "[red]✗[/]"
                    console.print(f"  {status} {r.query[:60]}... ({r.latency:.1f}s)")

            except FileNotFoundError:
                console.print(f"[error]File not found: {file_path}[/]")
            except Exception as e:
                console.print(f"[error]Batch failed: {e}[/]")

    elif cmd == "/fetch":
        tickers = args.strip().split()
        if not tickers:
            console.print("[warning]Usage: /fetch <TICKER> [TICKER2 ...][/]")
            console.print("[dim]  Example: /fetch AAPL NVDA GOOGL TSLA[/]")
        else:
            from src.rag_adapter import adapter

            for ticker in tickers:
                try:
                    with console.status(f"Fetching {ticker}..."):
                        count = adapter.fetch_and_add(ticker)
                    console.print(f"[green]✓[/] {ticker}: {count} chunks added to index")
                except Exception as e:
                    console.print(f"[red]✗[/] {ticker}: {e}")

    else:
        console.print(f"[warning]Unknown command: {cmd}. Type /help for available commands.[/]")

    return True


def _execute_query(question: str, chart_mode: bool = False) -> None:
    """Execute a query and display results."""
    from src.rag_adapter import adapter

    console.print(f"[dim]Querying: {question}[/]")
    start = time.time()

    try:
        result = adapter.query(question)
        latency = time.time() - start

        answer = result.get("model_answer", "No answer found.")

        # Extract confidence if present
        confidence = None
        if "[Confidence:" in answer:
            import re

            match = re.search(r"\[Confidence:\s*([\d.]+)\]", answer)
            if match:
                confidence = float(match.group(1))
                answer = re.sub(r"\s*\[Confidence:.*?\]", "", answer)

        print_answer(question, answer, confidence=confidence, latency=latency)

        # Update stats
        _session_stats["total_queries"] += 1
        _session_stats["total_latency"] += latency
        if confidence:
            _session_stats["confidences"].append(confidence)
        _session_stats["history"].append(
            {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "latency": latency,
            }
        )

    except Exception as e:
        console.print(f"[error]Query failed: {e}[/]")
        log_agent_action("CLI", "Error", str(e), level="error")


def main() -> None:
    """Main CLI entry point."""
    print_banner()

    # Startup health check
    health = check_ollama_health()
    print_status_dashboard(health, settings)

    if not health["healthy"]:
        console.print("[warning]⚠ Ollama is not available. Some features may not work.[/]")
        console.print(f"[dim]  Error: {health['error']}[/]")
        console.print("[dim]  Start Ollama: ollama serve[/]")
        console.print()

    console.print("[dim]Type /help for commands, or just ask a question. /quit to exit.[/]\n")

    # Interactive loop
    while True:
        try:
            user_input = Prompt.ask("[bold cyan]🏦 Swarm[/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye! 👋[/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            if not _handle_command(cmd, args):
                break
        else:
            _execute_query(user_input)


if __name__ == "__main__":
    main()
