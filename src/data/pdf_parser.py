"""
PDF to Markdown parser for financial documents.
Extracts text and tables from PDF files using PyMuPDF,
converts to structured Markdown for the RAG pipeline.

Usage:
    from src.data.pdf_parser import parse_pdf, parse_pdf_dir

    # Single PDF
    markdown = parse_pdf("data/raw/NVIDIA_10K_2025.pdf", company="NVIDIA")

    # Directory of PDFs
    combined = parse_pdf_dir("data/raw/", output_path="data/parsed/auto_parsed.md")
"""

import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
from src.utils.robustness import log_agent_action


def _extract_tables_from_page(page: fitz.Page) -> list[dict]:
    """Extract tables from a PDF page using PyMuPDF's table detection."""
    tables = []
    try:
        tab_finder = page.find_tables()
        for tab in tab_finder.tables:
            rows = tab.extract()
            if rows and len(rows) > 1:
                tables.append({
                    "headers": rows[0],
                    "rows": rows[1:],
                    "bbox": tab.bbox,
                })
    except Exception:
        pass
    return tables


def _table_to_markdown(table: dict) -> str:
    """Convert extracted table to Markdown format."""
    headers = table["headers"]
    rows = table["rows"]

    # Clean cells
    def clean(cell):
        if cell is None:
            return ""
        return str(cell).strip().replace("\n", " ")

    header_line = "| " + " | ".join(clean(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        # Pad row to match header length
        padded = list(row) + [""] * (len(headers) - len(row))
        row_lines.append("| " + " | ".join(clean(c) for c in padded[:len(headers)]) + " |")

    return "\n".join([header_line, sep_line] + row_lines)


def _extract_text_blocks(page: fitz.Page, table_bboxes: list) -> str:
    """Extract text blocks that don't overlap with tables."""
    blocks = page.get_text("blocks")
    text_parts = []

    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block[:7]
        if block_type != 0:  # Skip image blocks
            continue

        # Check if block overlaps with any table
        in_table = False
        for tbbox in table_bboxes:
            tx0, ty0, tx1, ty1 = tbbox
            if not (x1 < tx0 or x0 > tx1 or y1 < ty0 or y0 > ty1):
                in_table = True
                break

        if not in_table:
            clean_text = text.strip()
            if clean_text:
                text_parts.append(clean_text)

    return "\n\n".join(text_parts)


def _detect_section_headers(text: str) -> str:
    """Convert detected section headers to Markdown format."""
    lines = text.split("\n")
    result = []

    # Common financial document section patterns
    section_patterns = [
        r"^(CONSOLIDATED\s+(?:STATEMENTS?|BALANCE\s+SHEETS?).*)",
        r"^(REVENUE|INCOME|EXPENSES|ASSETS|LIABILITIES).*",
        r"^(Item\s+\d+[A-Z]?\.?\s+.+)",
        r"^(PART\s+[IVX]+)",
        r"^(Notes?\s+to\s+.+)",
    ]

    for line in lines:
        stripped = line.strip()
        is_header = False
        for pattern in section_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                result.append(f"\n## {stripped}\n")
                is_header = True
                break
        if not is_header:
            # Detect ALL CAPS lines as potential headers
            if (stripped.isupper() and len(stripped) > 5 and len(stripped) < 100
                    and not stripped.startswith("|")):
                result.append(f"\n## {stripped.title()}\n")
            else:
                result.append(stripped)

    return "\n".join(result)


def parse_pdf(pdf_path: str | Path, company: Optional[str] = None) -> str:
    """
    Parse a single PDF file into structured Markdown.

    Args:
        pdf_path: Path to PDF file
        company: Company name (auto-detected if not provided)

    Returns:
        Markdown string with headers, text, and tables
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    log_agent_action("PDFParser", "Open", f"{pdf_path.name}: {len(doc)} pages")

    # Auto-detect company from filename
    if not company:
        name = pdf_path.stem.upper()
        for c in ["NVIDIA", "APPLE", "MICROSOFT", "AAPL", "MSFT", "NVDA"]:
            if c in name:
                company = {"AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA"}.get(c, c.title())
                break
        if not company:
            company = pdf_path.stem.split("_")[0].title()

    sections = []
    sections.append(f"# {company} — {pdf_path.stem}\n")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract tables first
        tables = _extract_tables_from_page(page)
        table_bboxes = [t["bbox"] for t in tables]

        # Extract non-table text
        text = _extract_text_blocks(page, table_bboxes)

        if text.strip():
            processed = _detect_section_headers(text)
            sections.append(processed)

        # Add tables as Markdown
        for table in tables:
            if len(table["rows"]) > 0:
                md_table = _table_to_markdown(table)
                sections.append(f"\n{md_table}\n")

    doc.close()

    result = "\n\n".join(sections)
    log_agent_action("PDFParser", "Done", f"{pdf_path.name}: {len(result)} chars")
    return result


def parse_pdf_dir(
    pdf_dir: str | Path,
    output_path: Optional[str | Path] = None,
    company_map: Optional[dict] = None,
) -> str:
    """
    Parse all PDFs in a directory into a single Markdown file.

    Args:
        pdf_dir: Directory containing PDF files
        output_path: Optional path to save combined Markdown
        company_map: Optional {filename_pattern: company_name} mapping

    Returns:
        Combined Markdown string
    """
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    log_agent_action("PDFParser", "Batch", f"Found {len(pdfs)} PDFs in {pdf_dir}")

    all_markdown = []
    for pdf_path in pdfs:
        company = None
        if company_map:
            for pattern, name in company_map.items():
                if pattern.lower() in pdf_path.name.lower():
                    company = name
                    break

        try:
            md = parse_pdf(pdf_path, company=company)
            all_markdown.append(md)
        except Exception as e:
            log_agent_action("PDFParser", "Error", f"{pdf_path.name}: {e}")

    combined = "\n\n---\n\n".join(all_markdown)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined)
        log_agent_action("PDFParser", "Saved", f"{output_path}: {len(combined)} chars")

    return combined
