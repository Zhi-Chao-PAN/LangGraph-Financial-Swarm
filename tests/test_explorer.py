"""Tests for data explorer utilities."""

from src.ui.explorer import _count_tables, _extract_sections, _parse_companies


class TestParseCompanies:
    def test_basic(self):
        text = "# NVIDIA — Financial Data\n\nRevenue: $130B\n\n# Apple — Financial Data\n\nRevenue: $391B"
        companies = _parse_companies(text)
        assert "NVIDIA" in companies
        assert "Apple" in companies
        assert len(companies) == 2

    def test_single_company(self):
        text = "# Microsoft — Annual Report\n\nNet income: $88B"
        companies = _parse_companies(text)
        assert "Microsoft" in companies
        assert len(companies) == 1

    def test_empty(self):
        companies = _parse_companies("")
        assert len(companies) == 0


class TestCountTables:
    def test_with_tables(self):
        text = "| Year | Revenue |\n| 2024 | $100B |\n| 2025 | $130B |\n\nSome text\n\n| Metric | Value |\n| Margin | 75% |"
        assert _count_tables(text) == 2

    def test_no_tables(self):
        assert _count_tables("Just plain text here.") == 0

    def test_single_table(self):
        text = "| A | B |\n| 1 | 2 |"
        assert _count_tables(text) == 1


class TestExtractSections:
    def test_basic(self):
        text = "# NVIDIA\n\n## Revenue\n\nRevenue was $130B.\n\n## Margins\n\nGross margin 75%."
        sections = _extract_sections(text)
        assert len(sections) == 2
        assert sections[0]["title"] == "Revenue"
        assert sections[1]["title"] == "Margins"

    def test_table_detection(self):
        text = "# NVIDIA\n\n## Revenue\n\n| Year | Rev |\n| 2025 | $130B |\n\n## Notes\n\nSome text."
        sections = _extract_sections(text)
        assert sections[0]["has_table"]
        assert not sections[1]["has_table"]

    def test_char_count(self):
        text = "# NVIDIA\n\n## Revenue\n\nShort."
        sections = _extract_sections(text)
        assert sections[0]["char_count"] > 0

    def test_no_sections(self):
        text = "# NVIDIA\n\nJust a header, no sections."
        sections = _extract_sections(text)
        assert len(sections) == 0
