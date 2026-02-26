"""
Data source abstraction layer.
Provides a plugin-style architecture for multiple data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llama_index.core.schema import Document


@dataclass
class SourceMeta:
    """Metadata about a data source."""

    name: str
    description: str
    source_type: str  # "local", "api", "web"
    requires_api_key: bool = False


class DataSource(ABC):
    """Abstract base class for all data sources."""

    @abstractmethod
    def get_meta(self) -> SourceMeta:
        """Return metadata about this data source."""

    @abstractmethod
    def fetch(self, query: str, **kwargs) -> list[Document]:
        """Fetch documents matching the query."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is currently available."""


class LocalMarkdownSource(DataSource):
    """Data source for local Markdown files."""

    def __init__(self, file_path: str):
        self._path = file_path

    def get_meta(self) -> SourceMeta:
        return SourceMeta(
            name="Local Markdown",
            description="Pre-parsed financial data in Markdown format",
            source_type="local",
        )

    def fetch(self, query: str, **kwargs) -> list[Document]:
        import os

        from src.rag_adapter import _parse_sections

        if not os.path.exists(self._path):
            return []
        with open(self._path, encoding="utf-8") as f:
            text = f.read()
        return _parse_sections(text)

    def is_available(self) -> bool:
        import os

        return os.path.exists(self._path)


class YahooFinanceSource(DataSource):
    """Data source for Yahoo Finance real-time data."""

    def get_meta(self) -> SourceMeta:
        return SourceMeta(
            name="Yahoo Finance",
            description="Real-time financial data for any publicly traded company",
            source_type="api",
            requires_api_key=False,
        )

    def fetch(self, query: str, **kwargs) -> list[Document]:
        from src.data.finance_fetcher import fetch_and_convert
        from src.rag_adapter import _parse_sections

        ticker = kwargs.get("ticker", query)
        markdown = fetch_and_convert(ticker)
        return _parse_sections(markdown)

    def is_available(self) -> bool:
        try:
            import yfinance  # noqa: F401

            return True
        except ImportError:
            return False


@dataclass
class SourceRegistry:
    """Registry for managing multiple data sources."""

    _sources: dict[str, DataSource] = field(default_factory=dict)

    def register(self, name: str, source: DataSource):
        """Register a data source."""
        self._sources[name] = source

    def unregister(self, name: str):
        """Remove a data source."""
        self._sources.pop(name, None)

    def get(self, name: str) -> DataSource | None:
        """Get a data source by name."""
        return self._sources.get(name)

    def list_sources(self) -> list[SourceMeta]:
        """List all registered data sources with metadata."""
        return [s.get_meta() for s in self._sources.values()]

    def list_available(self) -> list[str]:
        """List names of available data sources."""
        return [name for name, s in self._sources.items() if s.is_available()]

    def fetch_all(self, query: str, **kwargs) -> list[Document]:
        """Fetch from all available sources and combine results."""
        docs = []
        for _name, source in self._sources.items():
            if source.is_available():
                try:
                    docs.extend(source.fetch(query, **kwargs))
                except Exception:  # noqa: BLE001
                    continue
        return docs


# Global registry
registry = SourceRegistry()
