import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "supervisor": "bold purple",
    "researcher": "bold blue",
    "quant": "bold green",
    "system": "dim white"
})

console = Console(theme=custom_theme)


class Observability:
    """Handles system observability: Rich UI, Metrics, and Tracing."""

    _start_time = 0
    _step_times = []
    _agent_calls = {}

    @staticmethod
    def start_trace():
        Observability._start_time = time.time()
        Observability._step_times = []
        Observability._agent_calls = {}
        banner = "[bold cyan]Financial Swarm Orchestrator[/bold cyan]"
        console.print(Panel(banner, border_style="cyan"))

    @staticmethod
    def trace_agent(agent_name: str, content: str, metadata: dict[str, Any] | None = None):
        """Log an agent's action with rich formatting."""
        style = agent_name.lower()
        if style not in ("supervisor", "researcher", "quant"):
            style = "info"

        # Track metrics
        Observability._step_times.append(time.time())
        Observability._agent_calls[agent_name] = Observability._agent_calls.get(agent_name, 0) + 1

        tokens = len(content) // 4
        meta_text = f"[dim]~{tokens} tokens | {time.strftime('%H:%M:%S')}[/]"

        # Truncate long content for display
        display_content = content[:500] + "..." if len(content) > 500 else content

        panel = Panel(
            Text(display_content),
            title=f"[{style}]{agent_name}[/]",
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
    def final_report(total_steps: int):
        elapsed = time.time() - Observability._start_time
        calls = Observability._agent_calls

        console.print(f"\n[bold green]Done[/] in {elapsed:.1f}s, {total_steps} steps.")
        if calls:
            parts = [f"{k}: {v}" for k, v in sorted(calls.items())]
            console.print(f"  [dim]Agent calls: {', '.join(parts)}[/]")

    @staticmethod
    def error(message: str):
        console.print(f"[danger]ERROR: {message}[/]")
