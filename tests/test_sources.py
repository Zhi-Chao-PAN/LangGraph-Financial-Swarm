"""Tests for data source abstraction layer."""

from src.data.sources import (
    LocalMarkdownSource,
    SourceMeta,
    SourceRegistry,
    YahooFinanceSource,
)


class TestSourceMeta:
    def test_basic(self):
        meta = SourceMeta(name="Test", description="A test source", source_type="local")
        assert meta.name == "Test"
        assert not meta.requires_api_key

    def test_api_key_required(self):
        meta = SourceMeta(name="API", description="API source", source_type="api", requires_api_key=True)
        assert meta.requires_api_key


class TestLocalMarkdownSource:
    def test_meta(self):
        source = LocalMarkdownSource("/tmp/test.md")
        meta = source.get_meta()
        assert meta.source_type == "local"
        assert "Markdown" in meta.name

    def test_not_available(self):
        source = LocalMarkdownSource("/nonexistent/path.md")
        assert not source.is_available()


class TestYahooFinanceSource:
    def test_meta(self):
        source = YahooFinanceSource()
        meta = source.get_meta()
        assert meta.source_type == "api"
        assert not meta.requires_api_key

    def test_available(self):
        source = YahooFinanceSource()
        assert source.is_available()


class TestSourceRegistry:
    def test_register_and_list(self):
        reg = SourceRegistry()
        source = YahooFinanceSource()
        reg.register("yahoo", source)
        assert len(reg.list_sources()) == 1
        assert reg.list_sources()[0].name == "Yahoo Finance"

    def test_get(self):
        reg = SourceRegistry()
        source = YahooFinanceSource()
        reg.register("yahoo", source)
        assert reg.get("yahoo") is source
        assert reg.get("nonexistent") is None

    def test_unregister(self):
        reg = SourceRegistry()
        reg.register("yahoo", YahooFinanceSource())
        reg.unregister("yahoo")
        assert len(reg.list_sources()) == 0

    def test_list_available(self):
        reg = SourceRegistry()
        reg.register("yahoo", YahooFinanceSource())
        reg.register("local", LocalMarkdownSource("/nonexistent.md"))
        available = reg.list_available()
        assert "yahoo" in available
        assert "local" not in available

    def test_fetch_all_empty(self):
        reg = SourceRegistry()
        docs = reg.fetch_all("test query")
        assert docs == []

    def test_register_multiple(self):
        reg = SourceRegistry()
        reg.register("a", YahooFinanceSource())
        reg.register("b", YahooFinanceSource())
        assert len(reg.list_sources()) == 2
