"""Load bundled prompts without referring to a partially constructed class."""

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")


class Prompts:
    """Central repository for agent system prompts."""

    BASE_DIR = PROMPT_DIR
    _load = staticmethod(_load_prompt)
    SUPERVISOR_SYSTEM = _load_prompt("supervisor.txt")
    RESEARCHER_SYSTEM = _load_prompt("researcher.txt")
    QUANT_SYSTEM = _load_prompt("quant.txt")
