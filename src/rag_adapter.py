"""
RAG adapter with hybrid search (BM25 + vector), structure-aware chunking,
async query support, caching with TTL, and retry with backoff.

Hybrid search uses Reciprocal Rank Fusion (RRF) to combine BM25 lexical
matching with dense vector retrieval for better recall on financial queries.
"""

import asyncio
import os
import re
import time
from typing import Any

import diskcache as dc
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.core.llm_index import DirectOllamaLLM
from src.utils.robustness import async_retry_with_backoff, log_agent_action


def _parse_sections(text: str) -> list[Document]:
    """
    Parse financial markdown into structured chunks.
    Each section (with its tables) becomes a separate Document with metadata.
    Company name is extracted from # headers and injected into every chunk.
    """
    company_blocks = re.split(r'\n(?=# [A-Z])', text)
    documents = []

    for block in company_blocks:
        first_line = block.strip().split('\n')[0]
        company_match = re.match(r'^#\s+(.+?)(?:\s*[—\-–]|\s*$)', first_line)
        company_name = company_match.group(1).strip() if company_match else ""

        sections = re.split(r'\n## ', block)

        for i, section in enumerate(sections):
            if i == 0:
                if section.strip():
                    documents.append(Document(
                        text=section.strip(),
                        metadata={"company": company_name, "section": "header", "index": len(documents)}
                    ))
                continue

            section_text = "## " + section
            lines = section_text.split('\n')
            section_title = lines[0].replace('##', '').strip()
            company_prefix = f"[{company_name}] " if company_name else ""

            table_lines = []
            non_table_lines = []
            in_table = False

            for line in lines[1:]:
                if '|' in line and '--' not in line:
                    in_table = True
                    table_lines.append(line)
                elif in_table and '|' not in line:
                    if table_lines:
                        table_text = f"{company_prefix}## {section_title}\n\n" + '\n'.join(table_lines)
                        documents.append(Document(
                            text=table_text,
                            metadata={
                                "company": company_name,
                                "section": section_title,
                                "type": "table",
                                "index": len(documents),
                            }
                        ))
                        table_lines = []
                    in_table = False
                    if line.strip():
                        non_table_lines.append(line)
                elif '---' in line:
                    continue
                else:
                    if in_table:
                        table_lines.append(line)
                    elif line.strip():
                        non_table_lines.append(line)

            if table_lines:
                table_text = f"{company_prefix}## {section_title}\n\n" + '\n'.join(table_lines)
                documents.append(Document(
                    text=table_text,
                    metadata={
                        "company": company_name,
                        "section": section_title,
                        "type": "table",
                        "index": len(documents),
                    }
                ))

            if non_table_lines:
                nt_text = f"{company_prefix}## {section_title}\n\n" + '\n'.join(non_table_lines)
                documents.append(Document(
                    text=nt_text,
                    metadata={
                        "company": company_name,
                        "section": section_title,
                        "type": "text",
                        "index": len(documents),
                    }
                ))

    log_agent_action("RAGAdapter", "Parse", f"Created {len(documents)} structured chunks")
    return documents


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-z0-9]+(?:\.[0-9]+)?', text.lower())


def _reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
    Each ranking is a list of document indices ordered by relevance.
    Returns list of (doc_index, rrf_score) sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class BM25Retriever:
    """BM25 lexical retriever over parsed documents."""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents
        corpus = [_tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(corpus)
        log_agent_action("BM25", "Init", f"Indexed {len(documents)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> list[int]:
        """Return top_k document indices ranked by BM25 score."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:top_k]

    def get_document(self, idx: int) -> Document:
        return self._documents[idx]


class RAGAdapter:
    """
    Adapter for LlamaIndex RAG with hybrid search (BM25 + vector),
    structure-aware chunking, and caching.
    """

    def __init__(self, cache_ttl: int = 86400) -> None:
        self.cache = dc.Cache(settings.DATA_DIR / "rag_cache")
        self.cache_ttl = cache_ttl
        self.index = None
        self.query_engine = None
        self._documents: list[Document] = []
        self._bm25: BM25Retriever | None = None
        self._embed_model = None
        self._llm = None
        self._lock = asyncio.Lock()

    def _build_index(self):
        """Build vector index and BM25 index with structure-aware chunking."""
        log_agent_action("RAGAdapter", "Init", "Building hybrid index (vector + BM25)...")

        self._embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        self._llm = DirectOllamaLLM(
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            request_timeout=settings.LLM_TIMEOUT,
        )

        data_path = settings.RAG_DATA_PATH
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"RAG data not found at {data_path}")

        with open(data_path, encoding='utf-8') as f:
            raw_text = f.read()

        self._documents = _parse_sections(raw_text)

        # Vector index
        self.index = VectorStoreIndex.from_documents(
            self._documents,
            embed_model=self._embed_model,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=settings.RAG_TOP_K,
            llm=self._llm,
        )

        # BM25 index
        self._bm25 = BM25Retriever(self._documents)

        return self._llm

    def _hybrid_retrieve(self, question: str, top_k: int | None = None) -> list[Document]:
        """
        Hybrid retrieval: BM25 + vector, fused with RRF.
        Returns top_k documents ranked by combined score.
        """
        if top_k is None:
            top_k = settings.RAG_TOP_K

        # BM25 ranking
        bm25_ranking = self._bm25.retrieve(question, top_k=top_k * 2)

        # Vector ranking
        retriever = self.index.as_retriever(similarity_top_k=top_k * 2)
        vector_nodes = retriever.retrieve(question)

        # Map vector results to document indices
        vector_ranking = []
        for node in vector_nodes:
            idx = node.node.metadata.get("index")
            if idx is not None:
                vector_ranking.append(idx)

        # RRF fusion
        fused = _reciprocal_rank_fusion([bm25_ranking, vector_ranking])
        top_indices = [idx for idx, _ in fused[:top_k]]

        results = [self._documents[i] for i in top_indices if i < len(self._documents)]
        log_agent_action("RAGAdapter", "Hybrid", f"BM25={len(bm25_ranking)}, Vec={len(vector_ranking)}, Fused={len(results)}")
        return results

    def query(self, question: str) -> dict[str, Any]:
        """Synchronous query with hybrid retrieval."""
        cached = self.cache.get(question)
        if cached:
            return cached

        if self.query_engine is None:
            self._build_index()

        # Hybrid retrieve then synthesize
        hybrid_docs = self._hybrid_retrieve(question)

        if hybrid_docs:
            context = "\n\n".join(doc.text for doc in hybrid_docs)
            from langchain_core.messages import SystemMessage
            try:
                prompt = (
                    f"Answer the following financial question using ONLY the provided context. "
                    f"Include specific numbers and cite the source sections.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
                response = self._llm.complete(prompt)
                answer_text = str(response).strip()
                answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()
            except Exception as e:
                log_agent_action("RAGAdapter", "Hybrid LLM Error", str(e)[:100], level="warning")
                # Fallback to standard query engine
                response = self.query_engine.query(question)
                result = self._format_result(response, question)
                self.cache.set(question, result, expire=self.cache_ttl)
                return result

            sources = []
            for doc in hybrid_docs:
                section = doc.metadata.get("section", "unknown")
                content = doc.text[:150]
                sources.append(f"[{section}]: {content}...")

            citation_str = "\n".join([f"[Source {i+1}] {s}" for i, s in enumerate(sources)])
            result = {
                "model_answer": answer_text + f"\n\n**Citations**:\n{citation_str}",
                "source_nodes": [doc.text for doc in hybrid_docs],
                "latency_s": 0.0,
            }
        else:
            response = self.query_engine.query(question)
            result = self._format_result(response, question)

        self.cache.set(question, result, expire=self.cache_ttl)
        return result

    async def _ensure_initialized(self):
        """Lazy async initialization."""
        if self.query_engine is not None:
            return
        async with self._lock:
            if self.query_engine is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._build_index)

    async def aquery(self, question: str) -> dict[str, Any]:
        """Async query with hybrid retrieval, caching and retry."""
        loop = asyncio.get_running_loop()

        cached = await loop.run_in_executor(None, lambda: self.cache.get(question))
        if cached:
            log_agent_action("RAGAdapter", "Cache Hit", f"Q: {question}")
            return cached

        await self._ensure_initialized()

        @async_retry_with_backoff(retries=3)
        async def _execute():
            return await loop.run_in_executor(None, lambda: self.query(question))

        try:
            start = time.time()
            result = await _execute()
            result["latency_s"] = time.time() - start
            return result
        except Exception as e:
            msg = f"RAG query failed: {e}"
            log_agent_action("RAGAdapter", "Error", msg)
            return {"model_answer": f"Error: {msg}", "source_nodes": [], "latency_s": 0.0}

    @staticmethod
    def _format_result(response, question: str, latency: float = 0.0) -> dict[str, Any]:
        sources = []
        if hasattr(response, "source_nodes"):
            for node in response.source_nodes:
                section = node.node.metadata.get("section", "unknown")
                content = node.node.get_content()[:150]
                sources.append(f"[{section}]: {content}...")

        citation_str = "\n".join([f"[Source {i+1}] {s}" for i, s in enumerate(sources)])

        result = {
            "model_answer": str(response) + f"\n\n**Citations**:\n{citation_str}",
            "source_nodes": [
                node.node.get_content()
                for node in response.source_nodes
            ] if hasattr(response, "source_nodes") else [],
            "latency_s": latency,
        }
        log_agent_action("RAGAdapter", "Query", f"Q: {question}")
        return result


adapter = RAGAdapter()
