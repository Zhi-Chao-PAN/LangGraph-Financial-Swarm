# src/prompts.py

import os
from pathlib import Path

class Prompts:
    """Central repository for all agent system prompts (Loaded from files)."""
    
    BASE_DIR = Path(__file__).parent.parent / "prompts" 
    
    @staticmethod
    def _load(filename):
        try:
             # Ensure directory exists if clean clone
             if not Prompts.BASE_DIR.exists():
                 return "System Prompt Not Found."
             return (Prompts.BASE_DIR / filename).read_text(encoding="utf-8")
        except Exception:
             return "Error Loading Prompt."

    SUPERVISOR_SYSTEM = _load.__func__("supervisor.txt")
    RESEARCHER_SYSTEM = _load.__func__("researcher.txt")

    QUANT_SYSTEM = _load.__func__("quant.txt")

