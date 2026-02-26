"""
System observability: Rich UI tracing, metrics, and structured logging.
Provides real-time agent activity visualization in the terminal.
"""

import time
import uuid
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "dim cyan",
        "warning": "magenta",
        "danger": "bold red",
        "supervisor": "bold purple",
        "researcher": "bold blue",
        "quant": "bold green",
        "system": "dim white",
    }
)

console = Console(theme=custom_theme)


class Observability:
    """Handles system observability: Rich UI, Metrics, and Tracing."""

    _start_time: float = 0
    _step_times: list[float] = []
    _agent_calls: dict[str, int] = {}
    _agent_latencies: dict[str, list[float]] = {}
    _query_id: str = ""
    _errors: list[dict[str, Any]] = []

    @staticmethod
    def start_trace(query: str | None = None) -> str:
        """Start a new trace. Returns a unique query ID."""
        Observability._start_time = time.time()
        Observability._step_times = []
        Observability._agent_calls = {}
        Observability._agent_latencies = {}
        Observability._errors = []
        Observability._query_id = str(uuid.uuid4())[:8]

        banner = "[bold cyan]Financial Swarm Orchestrator[/bold cyan]"
        if query:
            banner += f"\n[dim]Query: {query[:100]}[/dim]"
        banner += f"\n[dim]Trace ID: {Observability._query_id}[/dim]"
        console.print(Panel(banner, border_style="cyan"))
        return Observability._query_id

    @staticmethod
    def trace_agent(agent_name: str, content: str, metadata: dict[str, Any] | None = None):
        """Log an agent's action with rich formatting and timing."""
        style = agent_name.lower()
        if style not in ("supervisor", "researcher", "quant"):
            style = "info"

        now = time.time()
        step_latency = now - (Observability._step_times[-1] if Observability._step_times else Observability._start_time)
        Observability._step_times.append(now)

        # Track metrics
        Observability._agent_calls[agent_name] = Observability._agent_calls.get(agent_name, 0) + 1
        if agent_name not in Observability._agent_latencies:
            Observability._agent_latencies[agent_name] = []
        Observability._agent_latencies[agent_name].append(step_latency)

        tokens = len(content) // 4
        meta_parts = [
            f"~{tokens} tokens",
            f"{step_latency:.1f}s",
            time.strftime("%H:%M:%S"),
        ]
        if metadata:
            for k, v in metadata.items():
                meta_parts.append(f"{k}={v}")
        meta_text = f"[dim]{' | '.join(meta_parts)}[/]"

        display_content = content[:500] + "..." if len(content) > 500 else content

        panel = Panel(
            Text(display_content),
            title=f"[{style}]{agent_name}[/] [{Observability._query_id}]",
            subtitle=meta_text,
            border_style=style,
            expand=False,
        )
        console.print(panel)

    @staticmethod
    def trace_tool(tool_name: str, args: str, result: str):
        """Log tool execution."""
        console.print(f"  [dim]Tool: {tool_name}({args[:80]})[/]")
        console.print(f"  [dim]  -> {result[:100]}[/]")

    @staticmethod
    def trace_error(agent_name: str, error: str, recoverable: bool = True):
        """Log an error with classification."""
        level = "warning" if recoverable else "danger"
        Observability._errors.append(
            {
                "agent": agent_name,
                "error": error,
                "recoverable": recoverable,
                "timestamp": time.time(),
            }
        )
        console.print(f"  [{level}]{agent_name} ERROR: {error[:200]}[/]")

    @staticmethod
    def final_report(total_steps: int):
        """Print a detailed final report with per-agent breakdown."""
        elapsed = time.time() - Observability._start_time
        calls = Observability._agent_calls
        latencies = Observability._agent_latencies

        table = Table(title=f"Trace Report [{Observability._query_id}]", border_style="green")
        table.add_column("Agent", style="bold")
        table.add_column("Calls", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Total Time", justify="right")

        for agent, count in sorted(calls.items()):
            agent_lats = latencies.get(agent, [])
            avg_lat = sum(agent_lats) / len(agent_lats) if agent_lats else 0
            total_lat = sum(agent_lats)
            table.add_row(agent, str(count), f"{avg_lat:.2f}s", f"{total_lat:.2f}s")

        console.print(table)
        console.print(f"\n[bold green]Done[/] in {elapsed:.1f}s, {total_steps} steps, {len(Observability._errors)} errors.")

    @staticmethod
    def error(message: str):
        console.print(f"[danger]ERROR: {message}[/]")

    @staticmethod
    def get_metrics() -> dict[str, Any]:
        """Get current trace metrics as a dict."""
        elapsed = time.time() - Observability._start_time if Observability._start_time else 0
        return {
            "query_id": Observability._query_id,
            "elapsed_s": round(elapsed, 3),
            "agent_calls": dict(Observability._agent_calls),
            "error_count": len(Observability._errors),
        }
