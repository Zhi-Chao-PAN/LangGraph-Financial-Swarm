"""
End-to-end PDF parsing test.
Parse Apple + Microsoft 10-K PDFs → merge with NVIDIA hand-crafted data → validate output.

Usage:
    python eval/test_pdf_e2e.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pdf_parser import parse_pdf

SEC_CACHE = Path(r"D:\structure-aware-rag-study\data\.sec_download_cache\sec-edgar-filings")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "parsed" / "pdf_auto"

# Map company → latest 10-K PDF
PDF_MAP = {
    "Apple": list(SEC_CACHE.glob("AAPL/10-K/*/primary-document.pdf")),
    "Microsoft": list(SEC_CACHE.glob("MSFT/10-K/*/primary-document.pdf")),
}


def find_latest_pdf(paths: list[Path]) -> Path | None:
    """Return the most recent PDF by directory name (filing number)."""
    if not paths:
        return None
    return sorted(paths, key=lambda p: p.parent.name)[-1]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for company, pdfs in PDF_MAP.items():
        latest = find_latest_pdf(pdfs)
        if not latest or not latest.exists():
            print(f"[SKIP] {company}: no PDF found")
            continue

        print(f"\n{'=' * 60}")
        print(f"Parsing {company} 10-K: {latest}")
        print(f"File size: {latest.stat().st_size / 1024:.0f} KB")

        md = parse_pdf(str(latest), company=company)

        out_path = OUTPUT_DIR / f"{company.lower()}_10k_auto.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Basic quality checks
        char_count = len(md)
        line_count = md.count("\n")
        table_count = md.count("| --- |")
        section_count = md.count("## ")

        results[company] = {
            "chars": char_count,
            "lines": line_count,
            "tables": table_count,
            "sections": section_count,
            "output": str(out_path),
        }

        print(f"Output: {out_path}")
        print(f"  Characters: {char_count:,}")
        print(f"  Lines: {line_count:,}")
        print(f"  Tables: {table_count}")
        print(f"  Sections: {section_count}")

        # Assertions
        assert char_count > 10000, f"{company}: too few characters ({char_count})"
        assert line_count > 100, f"{company}: too few lines ({line_count})"
        assert table_count > 0, f"{company}: no tables found"
        print("  [OK] All checks passed")

    # Merge with NVIDIA hand-crafted data
    nvidia_path = Path(__file__).parent.parent / "data" / "parsed" / "llamaparse" / "parsed.md"
    if nvidia_path.exists():
        with open(nvidia_path, encoding="utf-8") as f:
            nvidia_data = f.read()

        # Extract NVIDIA sections only
        nvidia_sections = []
        in_nvidia = False
        for line in nvidia_data.split("\n"):
            if "NVIDIA" in line or "nvidia" in line.lower():
                in_nvidia = True
            if in_nvidia:
                nvidia_sections.append(line)

        merged_path = OUTPUT_DIR / "merged_all.md"
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write("# Merged Financial Data (Auto-parsed + Hand-crafted)\n\n")
            for company in ["Apple", "Microsoft"]:
                auto_path = OUTPUT_DIR / f"{company.lower()}_10k_auto.md"
                if auto_path.exists():
                    with open(auto_path, encoding="utf-8") as af:
                        f.write(af.read())
                    f.write("\n\n---\n\n")
            # Add NVIDIA from hand-crafted
            f.write("# NVIDIA (Hand-crafted data)\n\n")
            f.write("\n".join(nvidia_sections))

        merged_size = merged_path.stat().st_size
        print(f"\nMerged file: {merged_path} ({merged_size / 1024:.0f} KB)")

    print(f"\n{'=' * 60}")
    print("Summary:")
    for company, r in results.items():
        print(f"  {company}: {r['chars']:,} chars, {r['tables']} tables, {r['sections']} sections")
    print("E2E PDF test PASSED")


if __name__ == "__main__":
    main()
