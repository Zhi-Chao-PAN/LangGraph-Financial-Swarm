# src/data/ingest.py
import argparse
import os
from pathlib import Path

from src.core.config import settings


def ingest_data(input_path: str, output_path: str):
    """
    Ingest financial data for RAG. Supports PDF (via LlamaParse) and Markdown.
    """
    print(f"Ingesting: {input_path} -> {output_path}")

    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    api_key = settings.LLAMA_CLOUD_API_KEY

    if api_key and input_path.lower().endswith(".pdf"):
        return _ingest_pdf(input_path, output_path, api_key)
    if input_path.lower().endswith(".md"):
        return _ingest_markdown(input_path, output_path)
    print(f"Unsupported format: {input_path}")
    return False


def _ingest_pdf(input_path: str, output_path: str, api_key: str) -> bool:
    """Ingest PDF via LlamaParse."""
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

        print(f"PDF ingestion complete: {output_path}")
        return True
    except Exception as e:
        print(f"LlamaParse failed: {e}")
        return False


def _ingest_markdown(input_path: str, output_path: str) -> bool:
    """Copy markdown file to output path."""
    try:
        with open(input_path, encoding="utf-8") as f:
            content = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Markdown ingestion complete: {output_path}")
        return True
    except Exception as e:
        print(f"Ingestion failed: {e}")
        return False


def ingest_directory(input_dir: str, output_path: str):
    """Ingest all supported files from a directory into a single output."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.md")) + sorted(input_dir.glob("*.pdf"))

    if not files:
        print(f"No supported files in {input_dir}")
        return False

    combined = []
    for f in files:
        print(f"  Processing: {f.name}")
        if f.suffix == ".md":
            combined.append(f.read_text(encoding="utf-8"))
        # PDF would need LlamaParse, skip in batch mode

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n\n---\n\n".join(combined))

    print(f"Batch ingestion complete: {len(combined)} files -> {output_path}")
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
