"""
Streamlit data explorer page.
Allows users to browse loaded financial data, view chunks, and inspect metadata.

Usage:
    Import and call render_data_explorer() from the main Streamlit app.
"""

import re
from pathlib import Path

import streamlit as st


def _load_raw_data(rag_path: Path) -> str | None:
    """Load raw RAG data file."""
    if not rag_path.exists():
        return None
    return rag_path.read_text(encoding="utf-8")


def _parse_companies(text: str) -> dict[str, str]:
    """Parse company sections from the raw data."""
    companies = {}
    blocks = re.split(r"\n(?=# [A-Z])", text)
    for block in blocks:
        first_line = block.strip().split("\n")[0]
        match = re.match(r"^#\s+(.+?)(?:\s*[—\-–]|\s*$)", first_line)
        if match:
            name = match.group(1).strip()
            companies[name] = block.strip()
    return companies


def _count_tables(text: str) -> int:
    """Count markdown tables in text."""
    lines = text.split("\n")
    table_count = 0
    in_table = False
    for line in lines:
        if "|" in line and "---" not in line:
            if not in_table:
                table_count += 1
                in_table = True
        else:
            in_table = False
    return table_count


def _extract_sections(company_text: str) -> list[dict]:
    """Extract sections from a company block."""
    sections = []
    parts = re.split(r"\n## ", company_text)
    for i, part in enumerate(parts):
        if i == 0:
            continue
        lines = part.split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        has_table = "|" in content
        sections.append(
            {
                "title": title,
                "content": content,
                "has_table": has_table,
                "char_count": len(content),
            }
        )
    return sections


def render_data_explorer():
    """Render the data explorer page in Streamlit."""
    st.header("📊 Data Explorer")

    from src.core.config import settings

    rag_path = Path(settings.RAG_DATA_PATH)
    raw_data = _load_raw_data(rag_path)

    if not raw_data:
        st.warning(f"No RAG data found at {rag_path}")
        st.info("Run data ingestion first: `python -m src.data.ingest --input data/raw/ --output data/parsed/combined.md`")
        return

    # Overview metrics
    companies = _parse_companies(raw_data)
    total_tables = _count_tables(raw_data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies", len(companies))
    col2.metric("Total Size", f"{len(raw_data) / 1024:.0f} KB")
    col3.metric("Tables", total_tables)
    col4.metric("Lines", len(raw_data.split("\n")))

    st.divider()

    # Company selector
    if not companies:
        st.warning("No company data found in the RAG file.")
        return

    selected = st.selectbox("Select Company", list(companies.keys()))

    if selected:
        company_text = companies[selected]
        sections = _extract_sections(company_text)

        st.subheader(f"📁 {selected}")

        # Section overview
        sec_col1, sec_col2 = st.columns(2)
        sec_col1.metric("Sections", len(sections))
        sec_col2.metric("Tables", sum(1 for s in sections if s["has_table"]))

        # Section browser
        for sec in sections:
            icon = "📊" if sec["has_table"] else "📝"
            with st.expander(f"{icon} {sec['title']} ({sec['char_count']} chars)"):
                st.markdown(sec["content"][:2000])
                if len(sec["content"]) > 2000:
                    st.caption(f"... truncated ({sec['char_count']} total chars)")

    st.divider()

    # Raw data viewer
    with st.expander("🔍 Raw Data (first 5000 chars)"):
        st.code(raw_data[:5000], language="markdown")
