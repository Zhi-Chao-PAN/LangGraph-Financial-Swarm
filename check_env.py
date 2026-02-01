import sys
import shutil
import requests
from src.core.config import settings
from rich.console import Console

console = Console()

def check_environment():
    """Verify that all system dependencies are met."""
    console.rule("[bold green]System Environment Check[/]")
    
    # 1. Check Python Version
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 10:
        console.print("[green]✅ Python 3.10+ detected.[/]")
    else:
        console.print(f"[red]❌ Python version {sys.version} is not supported. Use 3.10+.[/]")
        sys.exit(1)

    # 2. Check Ollama
    try:
        resp = requests.get(f"{settings.LLM_BASE_URL}/api/tags")
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json()['models']]
            if settings.LLM_MODEL in models:
                console.print(f"[green]✅ Ollama is running and '{settings.LLM_MODEL}' is available.[/]")
            else:
                console.print(f"[yellow]⚠️  Ollama is running but model '{settings.LLM_MODEL}' not found. Run `ollama pull {settings.LLM_MODEL}`.[/]")
        else:
             console.print("[red]❌ Ollama API is not responding correctly.[/]")
    except requests.exceptions.ConnectionError:
        console.print("[red]❌ Could not connect to Ollama. Is it running?[/]")
        console.print("   Run: [bold]ollama serve[/]")

    # 3. Check Directories
    if settings.DATA_DIR.exists():
        console.print("[green]✅ Data directory exists.[/]")
    else:
        console.print("[yellow]⚠️  Data directory missing. Creating...[/]")
        settings.ensure_dirs()

    console.rule("[bold green]Check Complete[/]")

if __name__ == "__main__":
    check_environment()
