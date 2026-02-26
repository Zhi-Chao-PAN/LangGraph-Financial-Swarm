"""
Rich CLI interface for LangGraph Financial Swarm.
Provides a polished terminal UI with status dashboard, formatted output,
and interactive query mode.
"""

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "header": "bold magenta",
        "metric": "bold cyan",
        "company": "bold blue",
    }
)

console = Console(theme=custom_theme)


def print_banner() -> None:
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           🏦  LangGraph Financial Swarm  v0.9.0             ║
║     Multi-Agent Financial Analysis with Local LLMs          ║
╚══════════════════════════════════════════════════════════════╝"""
    console.print(banner, style="header")
    console.print()


def print_status_dashboard(health: dict, settings: Any) -> None:
    """Print system status dashboard on startup."""
    table = Table(title="System Status", show_header=True, header_style="bold cyan", border_style="blue")
    table.add_column("Component", style="bold", width=20)
    table.add_column("Status", width=15)
    table.add_column("Details", width=40)

    # Ollama status
    if health["healthy"]:
        table.add_row("Ollama", "[success]● Online[/]", f"Models: {', '.join(health['models'][:3])}")
    else:
        table.add_row("Ollama", "[error]● Offline[/]", health.get("error", "Unknown error"))

    # Model
    table.add_row("LLM Model", "[info]●[/]", settings.LLM_MODEL)

    # Embedding
    table.add_row("Embedding", "[info]●[/]", settings.EMBEDDING_MODEL)

    # RAG Data
    rag_path = Path(settings.RAG_DATA_PATH)
    if rag_path.exists():
        size_kb = rag_path.stat().st_size / 1024
        table.add_row("RAG Data", "[success]● Loaded[/]", f"{rag_path.name} ({size_kb:.0f} KB)")
    else:
        table.add_row("RAG Data", "[warning]● Missing[/]", str(rag_path))

    # Cache
    cache_dir = settings.DATA_DIR / "rag_cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.iterdir())
        table.add_row("Cache", "[info]●[/]", f"{len(cache_files)} entries")
    else:
        table.add_row("Cache", "[info]●[/]", "Empty")

    console.print(table)
    console.print()
