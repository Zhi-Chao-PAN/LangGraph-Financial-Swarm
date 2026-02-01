# src/rag_adapter.py (Updated with Citations & Robustness)
import os
import asyncio
from typing import Dict, Any, Optional
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from src.core.config import settings
from src.utils.robustness import retry_with_backoff, log_agent_action
import time
import diskcache as dc

class RAGAdapter:
    """
    Adapter for LlamaIndex RAG with persistent caching.
    """
    def __init__(self) -> None:
        # Cache setup
        self.cache = dc.Cache(settings.DATA_DIR / "rag_cache")
        
        # Configure LlamaIndex with local models (Validation)
        Settings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        Settings.llm = Ollama(model=settings.LLM_MODEL, base_url=settings.LLM_BASE_URL)
        
        self.index = self._load_data()
        self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    def _load_data(self) -> VectorStoreIndex:
        data_path = settings.RAG_DATA_PATH
        log_agent_action("RAGAdapter", "Initialization", "Loading/Parsing Index...")
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"RAG data not found at {data_path}. Please run the ingestion pipeline first.")
        
        documents = SimpleDirectoryReader(input_files=[data_path]).load_data()
        return VectorStoreIndex.from_documents(documents)

    @retry_with_backoff(retries=3)
    def ensure_initialized(self):
        """Ensures the index and query engine are loaded. (Now handled in __init__)"""
        # This method is largely vestigial now as index and query_engine are initialized in __init__
        # but kept for compatibility if external calls still rely on it.
        if self.index is None:
            self.index = self._load_data()
            self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    async def aquery(self, question: str) -> Dict[str, Any]:
        """
        Async query processing with Non-blocking Caching and Retry logic.
        """
        loop = asyncio.get_running_loop()
        
        # 1. Check Cache (Non-blocking)
        # DiskCache operations involve IO, so we offload to executor to prevent blocking the event loop
        cached_result = await loop.run_in_executor(None, lambda: self.cache.get(question))
        if cached_result:
            log_agent_action("RAGAdapter", "Query (Cache Hit)", f"Q: {question}")
            return cached_result # type: ignore

        start_time = time.time()
        
        @retry_with_backoff(retries=3)
        async def _execute_query():
            # Ensure initialization is verified (now strictly synchronous in __init__ or raises error)
            # In this strict mode, we assume __init__ succeeded or we fail fast.
            return await self.query_engine.aquery(question) 

        try:
            response = await _execute_query()
            
            # Extract citations
            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    sources.append(f"Content: {node.node.get_content()[:100]}...")
            
            citation_str = "\n".join([f"[Source {i+1}]: {s}" for i, s in enumerate(sources)])

            result = {
                "model_answer": str(response) + f"\n\n**Citations**:\n{citation_str}",
                "source_nodes": [node.node.get_content() for node in response.source_nodes] if hasattr(response, "source_nodes") else [],
                "latency_s": time.time() - start_time
            }
            
            log_agent_action("RAGAdapter", "Query", f"Q: {question} | A: {str(response)[:100]}...")
            
            # 2. Set Cache (Non-blocking)
            await loop.run_in_executor(None, lambda: self.cache.set(question, result))
            return result
            
        except Exception as e:
            # Fallback logic for robustness
            msg = f"RAG Engine Connection Failed: {e}. Please check your local Ollama/LlamaIndex setup."
            print(msg) # Keeping print here for immediate console visibility, logger used below
            log_agent_action("RAGAdapter", "Error", msg)
            return {
                "model_answer": f"Error: {msg}",
                "source_nodes": [],
                "latency_s": 0.0
            }


# Singleton instance for the tool to use
adapter = RAGAdapter()
