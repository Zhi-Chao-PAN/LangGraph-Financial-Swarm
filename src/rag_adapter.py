"""
RAG adapter with hybrid search (BM25 + vector), structure-aware chunking,
async query support, caching with TTL, and retry with backoff.

Hybrid search uses Reciprocal Rank Fusion (RRF) to combine BM25 lexical
matching with dense vector retrieval for better recall on financial queries.

SOTA enhancements:
- Cross-Encoder reranker for precision boost after RRF
- Sliding window context overlap between chunks
- Weighted RRF fusion (configurable BM25 vs vector weights)
- Adaptive Top-K based on query complexity
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
    company_blocks = re.split(r"\n(?=# [A-Z])", text)
    documents = []

    for block in company_blocks:
        first_line = block.strip().split("\n")[0]
        company_match = re.match(r"^#\s+(.+?)(?:\s*[—\-–]|\s*$)", first_line)
        company_name = company_match.group(1).strip() if company_match else ""

        sections = re.split(r"\n## ", block)

        for i, section in enumerate(sections):
            if i == 0:
                if section.strip():
                    documents.append(Document(text=section.strip(), metadata={"company": company_name, "section": "header", "index": len(documents)}))
                continue

            section_text = "## " + section
            lines = section_text.split("\n")
            section_title = lines[0].replace("##", "").strip()
            company_prefix = f"[{company_name}] " if company_name else ""

            table_lines = []
            non_table_lines = []
            in_table = False

            for line in lines[1:]:
                if "|" in line and "--" not in line:
                    in_table = True
                    table_lines.append(line)
                elif in_table and "|" not in line:
                    if table_lines:
                        table_text = f"{company_prefix}## {section_title}\n\n" + "\n".join(table_lines)
                        documents.append(
                            Document(
                                text=table_text,
                                metadata={
                                    "company": company_name,
                                    "section": section_title,
                                    "type": "table",
                                    "index": len(documents),
                                },
                            )
                        )
                        table_lines = []
                    in_table = False
                    if line.strip():
                        non_table_lines.append(line)
                elif "---" in line:
                    continue
                else:
                    if in_table:
                        table_lines.append(line)
                    elif line.strip():
                        non_table_lines.append(line)

            if table_lines:
                table_text = f"{company_prefix}## {section_title}\n\n" + "\n".join(table_lines)
                documents.append(
                    Document(
                        text=table_text,
                        metadata={
                            "company": company_name,
                            "section": section_title,
                            "type": "table",
                            "index": len(documents),
                        },
                    )
                )

            if non_table_lines:
                nt_text = f"{company_prefix}## {section_title}\n\n" + "\n".join(non_table_lines)
                documents.append(
                    Document(
                        text=nt_text,
                        metadata={
                            "company": company_name,
                            "section": section_title,
                            "type": "text",
                            "index": len(documents),
                        },
                    )
                )

    log_agent_action("RAGAdapter", "Parse", f"Created {len(documents)} structured chunks")
    return documents


def _add_context_overlap(documents: list[Document], overlap_lines: int = 3) -> list[Document]:
    """
    Add sliding window context overlap between adjacent chunks.
    Injects the last N lines of the previous chunk as prefix context
    to improve cross-section reference resolution.
    """
    if overlap_lines <= 0 or len(documents) < 2:
        return documents

    enhanced = [documents[0]]  # first chunk unchanged
    for i in range(1, len(documents)):
        prev_text = documents[i - 1].text
        prev_lines = prev_text.strip().split("\n")
        overlap = "\n".join(prev_lines[-overlap_lines:])

        # Only add overlap if chunks are from the same company
        same_company = (
            documents[i].metadata.get("company", "")
            == documents[i - 1].metadata.get("company", "")
        )

        if same_company and overlap.strip():
            new_text = f"[Context]\n{overlap}\n\n{documents[i].text}"
            new_meta = dict(documents[i].metadata)
            new_meta["has_overlap"] = True
            enhanced.append(Document(text=new_text, metadata=new_meta))
        else:
            enhanced.append(documents[i])

    return enhanced


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def _reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.
    Each ranking is a list of document indices ordered by relevance.
    Supports per-ranking weights for configurable BM25 vs vector emphasis.
    Returns list of (doc_index, rrf_score) sorted by score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings, strict=False):
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + w / (k + rank + 1)
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


def _compute_adaptive_top_k(question: str) -> int:
    """
    Dynamically adjust retrieval depth based on query complexity.
    Simple lookups need fewer chunks; multi-entity comparisons need more.
    """
    q_lower = question.lower()

    # Multi-entity/comparison queries need more context
    multi_indicators = [
        "compare", "vs", "versus", "which company", "highest",
        "lowest", "between", "rank", "all five", "all three",
        "respectively",
    ]
    multi_count = sum(1 for ind in multi_indicators if ind in q_lower)

    # Trend/time-range queries need more context
    trend_indicators = [
        "trend", "cagr", "growth", "from fy", "to fy",
        "over time", "year over year",
    ]
    trend_count = sum(1 for ind in trend_indicators if ind in q_lower)

    # Count mentioned companies
    companies = ["nvidia", "apple", "microsoft", "alphabet", "google", "amazon"]
    company_count = sum(1 for c in companies if c in q_lower)

    # Base top-k
    top_k = settings.RAG_TOP_K

    # Scale up for complexity
    if multi_count >= 2 or company_count >= 3:
        top_k = settings.RAG_TOP_K_MAX
    elif multi_count >= 1 or company_count >= 2:
        top_k = min(settings.RAG_TOP_K + 5, settings.RAG_TOP_K_MAX)
    elif trend_count >= 1:
        top_k = min(settings.RAG_TOP_K + 3, settings.RAG_TOP_K_MAX)

    # Simple single-entity lookups can use fewer
    if multi_count == 0 and trend_count == 0 and company_count <= 1:
        top_k = max(settings.RAG_TOP_K_MIN, settings.RAG_TOP_K - 1)

    return top_k


def _detect_query_type(question: str) -> str:
    """
    Classify query type for weighted RRF fusion.
    'numeric' queries benefit from BM25; 'semantic' from vector.
    """
    q_lower = question.lower()
    numeric_indicators = [
        "revenue", "income", "margin", "eps", "cash flow",
        "how much", "what was", "calculate", "ratio", "fy20",
        "spending", "assets", "debt", "equity",
    ]
    if sum(1 for ind in numeric_indicators if ind in q_lower) >= 2:
        return "numeric"
    return "semantic"


class RAGAdapter:
    """
    Adapter for LlamaIndex RAG with hybrid search (BM25 + vector),
    structure-aware chunking, cross-encoder reranking, and caching.
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
        self._reranker = None  # Lazy-loaded cross-encoder

    def _get_index_persist_dir(self) -> str:
        """Get the directory for persisted vector index."""
        return str(settings.DATA_DIR / "vector_index")

    def _build_index(self):
        """Build vector index and BM25 index with structure-aware chunking.
        Loads persisted index if available and data hasn't changed."""
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

        with open(data_path, encoding="utf-8") as f:
            raw_text = f.read()

        self._documents = _parse_sections(raw_text)

        # Add context overlap between adjacent chunks
        self._documents = _add_context_overlap(self._documents, overlap_lines=3)

        # Try loading persisted vector index
        persist_dir = self._get_index_persist_dir()
        data_hash_path = os.path.join(persist_dir, "data_hash.txt")
        import hashlib

        current_hash = hashlib.md5(raw_text.encode()).hexdigest()

        if os.path.exists(persist_dir) and os.path.exists(data_hash_path):
            try:
                with open(data_hash_path, encoding="utf-8") as f:
                    stored_hash = f.read().strip()
                if stored_hash == current_hash:
                    from llama_index.core import StorageContext, load_index_from_storage

                    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                    self.index = load_index_from_storage(storage_context, embed_model=self._embed_model)
                    self.query_engine = self.index.as_query_engine(
                        similarity_top_k=settings.RAG_TOP_K,
                        llm=self._llm,
                    )
                    self._bm25 = BM25Retriever(self._documents)
                    log_agent_action("RAGAdapter", "Init", "Loaded persisted vector index (cache hit)")
                    return self._llm
            except Exception as e:
                log_agent_action("RAGAdapter", "Init", f"Failed to load persisted index: {e}", level="warning")

        # Build fresh index
        self.index = VectorStoreIndex.from_documents(
            self._documents,
            embed_model=self._embed_model,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=settings.RAG_TOP_K,
            llm=self._llm,
        )

        # Persist index
        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.index.storage_context.persist(persist_dir=persist_dir)
            with open(data_hash_path, "w", encoding="utf-8") as f:
                f.write(current_hash)
            log_agent_action("RAGAdapter", "Init", "Vector index persisted to disk")
        except Exception as e:
            log_agent_action("RAGAdapter", "Init", f"Failed to persist index: {e}", level="warning")

        # BM25 index
        self._bm25 = BM25Retriever(self._documents)

        return self._llm

    def add_documents(self, new_docs: list[Document]) -> int:
        """
        Incrementally add documents to the index without full rebuild.
        Returns the number of documents added.
        """
        if not new_docs:
            return 0

        if self.index is None:
            self._build_index()

        # Add to vector index
        for doc in new_docs:
            self.index.insert(doc)

        # Update internal document list
        self._documents.extend(new_docs)

        # Rebuild BM25 (lightweight, in-memory)
        self._bm25 = BM25Retriever(self._documents)

        # Update query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=settings.RAG_TOP_K,
            llm=self._llm,
        )

        log_agent_action("RAGAdapter", "Incremental", f"Added {len(new_docs)} docs (total: {len(self._documents)})")
        return len(new_docs)

    def fetch_and_add(self, ticker: str) -> int:
        """
        Fetch financial data for a ticker via Yahoo Finance
        and add it to the index dynamically.
        Returns the number of chunks added.
        """
        from src.data.finance_fetcher import fetch_and_convert

        log_agent_action("RAGAdapter", "DynamicFetch", f"Fetching {ticker}...")
        markdown = fetch_and_convert(ticker)
        new_docs = _parse_sections(markdown)

        if not new_docs:
            log_agent_action("RAGAdapter", "DynamicFetch", f"No data for {ticker}", level="warning")
            return 0

        return self.add_documents(new_docs)

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker model."""
        if self._reranker is not None:
            return self._reranker
        if not settings.RERANKER_ENABLED:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.RERANKER_MODEL, max_length=512)
            log_agent_action("RAGAdapter", "Reranker", f"Loaded {settings.RERANKER_MODEL}")
            return self._reranker
        except ImportError:
            log_agent_action("RAGAdapter", "Reranker", "sentence-transformers not installed", level="warning")
            return None
        except Exception as e:
            log_agent_action("RAGAdapter", "Reranker", f"Failed to load: {e}", level="warning")
            return None

    def _hybrid_retrieve(self, question: str, top_k: int | None = None) -> list[Document]:
        """
        Hybrid retrieval: BM25 + vector, fused with weighted RRF.
        Optionally applies cross-encoder reranking for precision.
        Uses adaptive top-k based on query complexity.
        Returns top_k documents ranked by combined/reranker score.
        """
        if top_k is None:
            top_k = _compute_adaptive_top_k(question)

        # Determine RRF weights based on query type
        query_type = _detect_query_type(question)
        if query_type == "numeric":
            rrf_weights = [1.2, 0.8]  # Favor BM25 for numeric queries
        else:
            rrf_weights = [0.8, 1.2]  # Favor vector for semantic queries

        # BM25 ranking
        bm25_ranking = self._bm25.retrieve(question, top_k=top_k * 2)

        # Vector ranking
        retriever = self.index.as_retriever(similarity_top_k=top_k * 2)
        vector_nodes = retriever.retrieve(question)

        # Map vector results to document indices + collect scores
        vector_ranking = []
        vector_scores: dict[int, float] = {}
        for node in vector_nodes:
            idx = node.node.metadata.get("index")
            if idx is not None:
                vector_ranking.append(idx)
                vector_scores[idx] = node.score if hasattr(node, "score") and node.score else 0.0

        # Weighted RRF fusion
        fused = _reciprocal_rank_fusion(
            [bm25_ranking, vector_ranking],
            weights=rrf_weights,
        )
        top_indices = [idx for idx, _ in fused[:top_k * 2]]  # Get extra for reranking

        candidates = [self._documents[i] for i in top_indices if i < len(self._documents)]

        # Cross-encoder reranking (if enabled)
        reranker = self._get_reranker()
        if reranker and len(candidates) > 1:
            pairs = [(question, doc.text[:512]) for doc in candidates]
            try:
                rerank_scores = reranker.predict(pairs)
                scored = sorted(
                    zip(candidates, rerank_scores),
                    key=lambda x: float(x[1]),
                    reverse=True,
                )
                candidates = [doc for doc, _ in scored[:top_k]]
                log_agent_action(
                    "RAGAdapter", "Rerank",
                    f"Reranked {len(pairs)} candidates → top {len(candidates)}",
                )
            except Exception as e:
                log_agent_action("RAGAdapter", "Rerank", f"Failed: {e}", level="warning")
                candidates = candidates[:top_k]
        else:
            candidates = candidates[:top_k]

        # Collect retrieval scores for confidence propagation
        retrieval_scores = []
        for doc in candidates:
            idx = doc.metadata.get("index", -1)
            retrieval_scores.append(vector_scores.get(idx, 0.5))

        log_agent_action(
            "RAGAdapter", "Hybrid",
            f"BM25={len(bm25_ranking)}, Vec={len(vector_ranking)}, "
            f"Final={len(candidates)}, Type={query_type}",
        )
        return candidates, retrieval_scores

    def query(self, question: str) -> dict[str, Any]:
        """Synchronous query with hybrid retrieval."""
        cached = self.cache.get(question)
        if cached:
            return cached

        if self.query_engine is None:
            self._build_index()

        # Hybrid retrieve then synthesize
        hybrid_docs, retrieval_scores = self._hybrid_retrieve(question)

        if hybrid_docs:
            context = "\n\n".join(doc.text for doc in hybrid_docs)
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

            citation_str = "\n".join([f"[Source {i + 1}] {s}" for i, s in enumerate(sources)])
            result = {
                "model_answer": answer_text + f"\n\n**Citations**:\n{citation_str}",
                "source_nodes": [doc.text for doc in hybrid_docs],
                "retrieval_scores": retrieval_scores,
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

        citation_str = "\n".join([f"[Source {i + 1}] {s}" for i, s in enumerate(sources)])

        result = {
            "model_answer": str(response) + f"\n\n**Citations**:\n{citation_str}",
            "source_nodes": [node.node.get_content() for node in response.source_nodes] if hasattr(response, "source_nodes") else [],
            "latency_s": latency,
        }
        log_agent_action("RAGAdapter", "Query", f"Q: {question}")
        return result


adapter = RAGAdapter()
