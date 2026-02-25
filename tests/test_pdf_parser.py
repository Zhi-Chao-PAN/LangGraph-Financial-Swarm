"""Tests for PDF parser module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.pdf_parser import (
    _detect_section_headers,
    _table_to_markdown,
    parse_pdf,
)


class TestTableToMarkdown:
    def test_basic_table(self):
        table = {
            "headers": ["Metric", "FY2024", "FY2025"],
            "rows": [["Revenue", "60,922", "130,497"], ["Net Income", "29,760", "72,880"]],
            "bbox": (0, 0, 100, 100),
        }
        md = _table_to_markdown(table)
        assert "| Metric | FY2024 | FY2025 |" in md
        assert "| Revenue | 60,922 | 130,497 |" in md
        assert "| Net Income | 29,760 | 72,880 |" in md
        assert "| --- | --- | --- |" in md

    def test_none_cells(self):
        table = {
            "headers": ["A", "B"],
            "rows": [[None, "value"], ["x", None]],
            "bbox": (0, 0, 50, 50),
        }
        md = _table_to_markdown(table)
        assert "|  | value |" in md
        assert "| x |  |" in md

    def test_short_row_padded(self):
        table = {
            "headers": ["A", "B", "C"],
            "rows": [["only_one"]],
            "bbox": (0, 0, 50, 50),
        }
        md = _table_to_markdown(table)
        assert "| only_one |  |  |" in md

    def test_newline_in_cell(self):
        table = {
            "headers": ["Item"],
            "rows": [["line1\nline2"]],
            "bbox": (0, 0, 50, 50),
        }
        md = _table_to_markdown(table)
        assert "line1 line2" in md
        assert "\n" not in md.split("\n")[2]  # data row should not have embedded newline


class TestDetectSectionHeaders:
    def test_consolidated_statement(self):
        text = "CONSOLIDATED STATEMENTS OF INCOME\nRevenue was $100M."
        result = _detect_section_headers(text)
        assert "## CONSOLIDATED STATEMENTS OF INCOME" in result

    def test_all_caps_header(self):
        text = "REVENUE BREAKDOWN\nData Center: $115B"
        result = _detect_section_headers(text)
        assert "##" in result
        assert "REVENUE BREAKDOWN" in result or "Revenue Breakdown" in result

    def test_normal_text_unchanged(self):
        text = "The company reported strong growth."
        result = _detect_section_headers(text)
        assert "##" not in result
        assert "The company reported strong growth." in result

    def test_short_caps_not_header(self):
        text = "USD\nSome text"
        result = _detect_section_headers(text)
        # "USD" is too short (< 5 chars) to be detected as header
        assert "## " not in result or "Usd" not in result


class TestParsePdf:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_pdf("/nonexistent/path.pdf")

    def test_company_auto_detect_nvidia(self):
        # Test filename-based company detection
        with tempfile.NamedTemporaryFile(suffix="_NVIDIA_10K.pdf", delete=False) as f:
            tmp_path = f.name

        # Mock fitz.open to avoid needing a real PDF
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__iter__ = MagicMock(return_value=iter([]))
        mock_page = MagicMock()
        mock_page.find_tables.return_value = MagicMock(tables=[])
        mock_page.get_text.return_value = []
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("src.data.pdf_parser.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            mock_doc.__len__.return_value = 1

            # Create a range mock for iteration
            mock_doc.close = MagicMock()
            result = parse_pdf(tmp_path)
            assert "NVIDIA" in result or "Nvidia" in result

        Path(tmp_path).unlink(missing_ok=True)

    def test_company_auto_detect_apple(self):
        with tempfile.NamedTemporaryFile(suffix="_AAPL_10K.pdf", delete=False) as f:
            tmp_path = f.name

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=0)
        mock_doc.close = MagicMock()

        with patch("src.data.pdf_parser.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            result = parse_pdf(tmp_path, company="Apple")
            assert "Apple" in result

        Path(tmp_path).unlink(missing_ok=True)
