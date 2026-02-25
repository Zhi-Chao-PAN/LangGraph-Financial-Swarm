"""
Parse PDF financial documents into Markdown for RAG ingestion.

Usage:
    python -m src.data.pdf_parser --input data/raw/ --output data/parsed/auto_parsed.md
    python -m src.data.pdf_parser --input report.pdf --output parsed.md --company NVIDIA
"""

import argparse
from pathlib import Path
from src.data.pdf_parser import parse_pdf, parse_pdf_dir


def main():
    parser = argparse.ArgumentParser(description="PDF to Markdown parser for financial docs")
    parser.add_argument("--input", required=True, help="PDF file or directory")
    parser.add_argument("--output", required=True, help="Output Markdown file path")
    parser.add_argument("--company", default=None, help="Company name (auto-detected if omitted)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        result = parse_pdf_dir(input_path, output_path=args.output)
    elif input_path.is_file():
        result = parse_pdf(input_path, company=args.company)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(f"Error: {input_path} not found")
        return

    print(f"Done! Output: {args.output} ({len(result)} chars)")


if __name__ == "__main__":
    main()
