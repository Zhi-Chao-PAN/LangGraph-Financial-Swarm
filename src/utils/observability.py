import time
from typing import Dict, Any, Optional
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

# Define custom theme for agents
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

import uuid

class Observability:
    """
    Handles system observability: Rich UI, Metrics, and Tracing.
    """
    _start_time = 0
    _run_id = str(uuid.uuid4())[:8]
    
    @staticmethod
    def start_trace():
        Observability._start_time = time.time()
        banner = """
[bold cyan]
███████╗██╗███╗   ██╗ █████╗ ███╗   ██╗ ██████╗██╗ █████╗ ██╗     
██╔════╝██║████╗  ██║██╔══██╗████╗  ██║██╔════╝██║██╔══██╗██║     
█████╗  ██║██╔██╗ ██║███████║██╔██╗ ██║██║     ██║███████║██║     
██╔══╝  ██║██║╚██╗██║██╔══██║██║╚██╗██║██║     ██║██╔══██║██║     
██║     ██║██║ ╚████║██║  ██║██║ ╚████║╚██████╗██║██║  ██║███████╗
╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝
           S W A R M   O R C H E S T R A T O R
[/bold cyan]
        """
        console.print(banner)
        console.print(Panel(f"[bold green]Financial Swarm System Online (Run ID: {Observability._run_id})[/]", border_style="cyan"))

    @staticmethod
    def trace_agent(agent_name: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log an agent's action with rich formatting.
        """
        style = agent_name.lower()
        if style not in ["supervisor", "researcher", "quant"]:
            style = "info"
            
        # Estimate tokens (Heuristic: 1 token ~= 4 chars)
        tokens = len(content) // 4
        
        # Create metadata text
        meta_text = f"[dim]Tokens: ~{tokens} | Time: {time.strftime('%H:%M:%S')}[/]"
        if metadata:
            for k, v in metadata.items():
                meta_text += f", {k}: {v}"
                
        panel = Panel(
            Text(content),
            title=f"[{style}]{agent_name}[/]",
            subtitle=meta_text,
            border_style=style,
            expand=False
        )
        console.print(panel)

    @staticmethod
    def trace_tool(tool_name: str, args: str, result: str):
        """Log tool execution."""
        console.print(f"  [dim]🛠️  Tool Call: {tool_name}({args})[/]")
        console.print(f"  [dim]   -> Result: {result[:100]}...[/]")

    @staticmethod
    def final_report(total_steps: int):
        elapsed = time.time() - Observability._start_time
        console.print(f"\n[bold green]Mission Complete[/] in {elapsed:.2f}s ({total_steps} steps).")

    @staticmethod
    def error(message: str):
        console.print(f"[danger]ERROR: {message}[/]")
