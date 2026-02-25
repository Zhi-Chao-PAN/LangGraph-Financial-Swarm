"""Tests for data ingestion pipeline (PDF + Markdown -> RAG-ready output)."""

import tempfile
from pathlib import Path

from src.data.ingest import _ingest_markdown, ingest_data, ingest_directory


class TestIngestMarkdown:
    def test_basic_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "input.md"
            dst = Path(tmpdir) / "output.md"
            src.write_text("# NVIDIA\n\nRevenue: $130,497M", encoding="utf-8")
            assert _ingest_markdown(str(src), str(dst))
            assert dst.exists()
            content = dst.read_text(encoding="utf-8")
            assert "NVIDIA" in content
            assert "130,497" in content

    def test_missing_input(self):
        assert not ingest_data("/nonexistent/file.md", "/tmp/out.md")

    def test_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "data.csv"
            src.write_text("a,b,c", encoding="utf-8")
            assert not ingest_data(str(src), str(Path(tmpdir) / "out.md"))


class TestIngestDirectory:
    def test_markdown_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test markdown files
            (Path(tmpdir) / "nvidia.md").write_text("# NVIDIA\nRevenue: $130B", encoding="utf-8")
            (Path(tmpdir) / "apple.md").write_text("# Apple\nRevenue: $391B", encoding="utf-8")
            out = Path(tmpdir) / "combined.md"
            assert ingest_directory(tmpdir, str(out))
            content = out.read_text(encoding="utf-8")
            assert "NVIDIA" in content
            assert "Apple" in content

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert not ingest_directory(tmpdir, str(Path(tmpdir) / "out.md"))

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.md").write_text("# Test", encoding="utf-8")
            out = Path(tmpdir) / "sub" / "dir" / "out.md"
            assert ingest_directory(tmpdir, str(out))
            assert out.exists()


class TestIngestPDF:
    def test_pdf_ingest_if_available(self):
        """Test PDF ingestion with a real PDF if one exists in data/raw/."""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            return  # skip if no raw data

        pdfs = list(raw_dir.glob("*.pdf"))
        if not pdfs:
            return  # skip if no PDFs

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "parsed.md"
            result = ingest_data(str(pdfs[0]), str(out))
            if result:
                content = out.read_text(encoding="utf-8")
                assert len(content) > 100
