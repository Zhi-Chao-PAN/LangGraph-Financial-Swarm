"""Tests for hybrid search (BM25 + vector RRF fusion)."""

from llama_index.core import Document

from src.rag_adapter import (
    BM25Retriever,
    _parse_sections,
    _reciprocal_rank_fusion,
    _tokenize,
)


# ─── Tokenizer ───

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("NVIDIA Revenue FY2025")
        assert "nvidia" in tokens
        assert "revenue" in tokens
        assert "fy2025" in tokens

    def test_numbers_preserved(self):
        tokens = _tokenize("$130,497M in revenue")
        assert "130" in tokens or "497" in tokens
        assert "revenue" in tokens

    def test_decimal(self):
        tokens = _tokenize("margin was 77.6%")
        assert "77.6" in tokens

    def test_empty(self):
        assert _tokenize("") == []

    def test_special_chars_stripped(self):
        tokens = _tokenize("revenue---growth!!!")
        assert "revenue" in tokens
        assert "growth" in tokens


# ─── Reciprocal Rank Fusion ───

class TestRRF:
    def test_single_ranking(self):
        result = _reciprocal_rank_fusion([[0, 1, 2]])
        indices = [idx for idx, _ in result]
        assert indices == [0, 1, 2]

    def test_two_rankings_agreement(self):
        """Both rankings agree -> same order, higher scores."""
        result = _reciprocal_rank_fusion([[0, 1, 2], [0, 1, 2]])
        indices = [idx for idx, _ in result]
        assert indices[0] == 0  # top doc agreed by both

    def test_two_rankings_disagreement(self):
        """Rankings disagree -> all docs present, scores reflect combined rank."""
        result = _reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]])
        indices = [idx for idx, _ in result]
        # All 3 docs should appear
        assert set(indices) == {0, 1, 2}
        # Doc 0 and 2 each have one rank-0 and one rank-2, so tied and above doc 1
        scores = {idx: s for idx, s in result}
        assert scores[0] == scores[2]
        assert scores[0] > scores[1]

    def test_disjoint_rankings(self):
        """No overlap -> all docs appear."""
        result = _reciprocal_rank_fusion([[0, 1], [2, 3]])
        indices = [idx for idx, _ in result]
        assert set(indices) == {0, 1, 2, 3}

    def test_empty_ranking(self):
        result = _reciprocal_rank_fusion([[]])
        assert result == []

    def test_scores_decrease(self):
        result = _reciprocal_rank_fusion([[0, 1, 2, 3]])
        scores = [s for _, s in result]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


# ─── BM25 Retriever ───

class TestBM25Retriever:
    def _make_docs(self) -> list[Document]:
        return [
            Document(text="NVIDIA revenue was $130,497 million in FY2025", metadata={"index": 0}),
            Document(text="Apple gross margin was 46.9% in FY2024", metadata={"index": 1}),
            Document(text="Microsoft net income reached $88,136 million", metadata={"index": 2}),
            Document(text="NVIDIA Data Center revenue grew from $15,005M to $115,199M", metadata={"index": 3}),
        ]

    def test_retrieve_nvidia(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        results = bm25.retrieve("NVIDIA revenue FY2025", top_k=2)
        assert len(results) == 2
        # NVIDIA revenue doc should be top
        assert 0 in results

    def test_retrieve_apple(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        results = bm25.retrieve("Apple gross margin", top_k=1)
        assert results[0] == 1

    def test_retrieve_microsoft(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        results = bm25.retrieve("Microsoft net income", top_k=1)
        assert results[0] == 2

    def test_top_k_limit(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        results = bm25.retrieve("revenue", top_k=2)
        assert len(results) == 2

    def test_empty_query(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        results = bm25.retrieve("", top_k=3)
        assert results == []

    def test_get_document(self):
        docs = self._make_docs()
        bm25 = BM25Retriever(docs)
        doc = bm25.get_document(1)
        assert "Apple" in doc.text


# ─── Parse Sections ───

class TestParseSections:
    def test_basic_parsing(self):
        text = """# NVIDIA — Financial Data

## Revenue

| Year | Revenue |
|------|---------|
| FY2025 | $130,497M |

## Margins

Gross margin was 75.0%.
"""
        docs = _parse_sections(text)
        assert len(docs) >= 2  # at least header + table + text

    def test_company_metadata(self):
        text = """# NVIDIA — Financial Data

## Revenue

Revenue was $130,497M.
"""
        docs = _parse_sections(text)
        companies = [d.metadata.get("company", "") for d in docs]
        assert any("NVIDIA" in c for c in companies)

    def test_table_vs_text_type(self):
        text = """# Apple — Financial Data

## Revenue

| Year | Revenue |
|------|---------|
| FY2024 | $391,035M |

## Notes

Apple had strong growth.
"""
        docs = _parse_sections(text)
        types = [d.metadata.get("type", "") for d in docs]
        assert "table" in types
        assert "text" in types
