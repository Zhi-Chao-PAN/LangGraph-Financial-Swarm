"""
RAG adapter with structure-aware chunking, async query support,
caching with TTL, and retry with backoff.
"""

import asyncio
import os
import re
import time
from typing import Any

import diskcache as dc
from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.core.config import settings
from src.core.llm_index import DirectOllamaLLM
from src.utils.robustness import async_retry_with_backoff, log_agent_action


def _parse_sections(text: str) -> list[Document]:
    """
    Parse financial markdown into structured chunks.
    Each section (with its tables) becomes a separate Document with metadata.
    Company name is extracted from # headers and injected into every chunk.
    """
    # Split by top-level company headers (# Company Name)
    company_blocks = re.split(r'\n(?=# [A-Z])', text)
    documents = []

    for block in company_blocks:
        # Extract company name from first line
        first_line = block.strip().split('\n')[0]
        company_match = re.match(r'^#\s+(.+?)(?:\s*[—\-–]|\s*$)', first_line)
        company_name = company_match.group(1).strip() if company_match else ""

        # Split into sections within this company block
        sections = re.split(r'\n## ', block)

        for i, section in enumerate(sections):
            if i == 0:
                # Header section
                if section.strip():
                    documents.append(Document(
                        text=section.strip(),
                        metadata={"company": company_name, "section": "header", "index": len(documents)}
                    ))
                continue

            section_text = "## " + section
            lines = section_text.split('\n')
            section_title = lines[0].replace('##', '').strip()

            # Prefix with company name for disambiguation
            company_prefix = f"[{company_name}] " if company_name else ""

            # Find tables within this section
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

            # Flush remaining table
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

            # Non-table text
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


class RAGAdapter:
    """
    Adapter for LlamaIndex RAG with structure-aware chunking and caching.
    """

    def __init__(self, cache_ttl: int = 86400) -> None:
        self.cache = dc.Cache(settings.DATA_DIR / "rag_cache")
        self.cache_ttl = cache_ttl  # seconds, default 24h
        self.index = None
        self.query_engine = None
        self._lock = asyncio.Lock()

    def _build_index(self):
        """Build index with structure-aware chunking."""
        log_agent_action("RAGAdapter", "Init", "Building structured index...")

        embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        llm = DirectOllamaLLM(
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

        # Structure-aware parsing
        documents = _parse_sections(raw_text)

        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=embed_model,
        )
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=settings.RAG_TOP_K,
            llm=llm,
        )
        return llm

    def query(self, question: str) -> dict[str, Any]:
        """Synchronous query interface."""
        cached = self.cache.get(question)
        if cached:
            return cached

        if self.query_engine is None:
            self._build_index()

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
        """Async query with caching and retry."""
        loop = asyncio.get_running_loop()

        cached = await loop.run_in_executor(None, lambda: self.cache.get(question))
        if cached:
            log_agent_action("RAGAdapter", "Cache Hit", f"Q: {question}")
            return cached

        await self._ensure_initialized()

        @async_retry_with_backoff(retries=3)
        async def _execute():
            return await self.query_engine.aquery(question)

        try:
            start = time.time()
            response = await _execute()
            result = self._format_result(response, question, time.time() - start)
            await loop.run_in_executor(None, lambda: self.cache.set(question, result, expire=self.cache_ttl))
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
