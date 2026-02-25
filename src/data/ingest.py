"""
Data ingestion pipeline for financial documents.
Supports PDF (via PyMuPDF local parser or LlamaParse cloud) and Markdown.

Usage:
    python -m src.data.ingest --input data/raw/NVIDIA_10K.pdf --output data/financials.md
    python -m src.data.ingest --input data/raw/ --output data/financials.md
"""

import argparse
import os
from pathlib import Path

from src.core.config import settings
from src.data.pdf_parser import parse_pdf
from src.utils.robustness import log_agent_action


def ingest_data(input_path: str, output_path: str) -> bool:
    """
    Ingest financial data for RAG. Supports PDF and Markdown.
    PDF parsing uses local PyMuPDF by default; falls back to LlamaParse if API key set.
    """
    log_agent_action("Ingest", "Start", f"{input_path} -> {output_path}")

    if not os.path.exists(input_path):
        log_agent_action("Ingest", "Error", f"{input_path} does not exist.")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if input_path.lower().endswith(".pdf"):
        return _ingest_pdf(input_path, output_path)
    if input_path.lower().endswith(".md"):
        return _ingest_markdown(input_path, output_path)

    log_agent_action("Ingest", "Error", f"Unsupported format: {input_path}")
    return False


def _ingest_pdf(input_path: str, output_path: str) -> bool:
    """Ingest PDF: local PyMuPDF first, LlamaParse as fallback."""
    # Try local PyMuPDF parser first (no API key needed)
    try:
        markdown = parse_pdf(input_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        log_agent_action("Ingest", "Done", f"PDF (local): {len(markdown)} chars -> {output_path}")
        return True
    except Exception as e:
        log_agent_action("Ingest", "Warning", f"Local PDF parse failed: {e}")

    # Fallback to LlamaParse if API key available
    api_key = settings.LLAMA_CLOUD_API_KEY
    if api_key:
        try:
            from llama_parse import LlamaParse

            parser = LlamaParse(
                api_key=api_key,
                result_type="markdown",
                verbose=True,
                language="en",
                num_workers=4,
            )
            documents = parser.load_data(input_path)
            content = "\n\n".join(doc.text for doc in documents)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            log_agent_action("Ingest", "Done", f"PDF (LlamaParse): {output_path}")
            return True
        except Exception as e:
            log_agent_action("Ingest", "Error", f"LlamaParse failed: {e}")
            return False

    log_agent_action("Ingest", "Error", "PDF parsing failed and no LlamaParse API key")
    return False


def _ingest_markdown(input_path: str, output_path: str) -> bool:
    """Copy markdown file to output path."""
    try:
        with open(input_path, encoding="utf-8") as f:
            content = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        log_agent_action("Ingest", "Done", f"Markdown: {len(content)} chars -> {output_path}")
        return True
    except Exception as e:
        log_agent_action("Ingest", "Error", f"Ingestion failed: {e}")
        return False


def ingest_directory(input_dir: str, output_path: str) -> bool:
    """Ingest all supported files from a directory into a single output."""
    input_dir_path = Path(input_dir)
    md_files = sorted(input_dir_path.glob("*.md"))
    pdf_files = sorted(input_dir_path.glob("*.pdf"))

    if not md_files and not pdf_files:
        log_agent_action("Ingest", "Error", f"No supported files in {input_dir}")
        return False

    combined = []

    # Process markdown files
    for f in md_files:
        log_agent_action("Ingest", "Process", f.name)
        combined.append(f.read_text(encoding="utf-8"))

    # Process PDF files with local parser
    for f in pdf_files:
        log_agent_action("Ingest", "Process", f"{f.name} (PDF)")
        try:
            md = parse_pdf(f)
            combined.append(md)
        except Exception as e:
            log_agent_action("Ingest", "Warning", f"Skipped {f.name}: {e}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n\n---\n\n".join(combined))

    log_agent_action("Ingest", "Done", f"Batch: {len(combined)} files -> {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial data for RAG.")
    parser.add_argument("--input", type=str, help="Path to raw data (PDF/MD) or directory")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    input_file = args.input or "data/raw/sample_financials.md"
    output_file = args.output or str(settings.RAG_DATA_PATH)

    if os.path.isdir(input_file):
        ingest_directory(input_file, output_file)
    else:
        ingest_data(input_file, output_file)
