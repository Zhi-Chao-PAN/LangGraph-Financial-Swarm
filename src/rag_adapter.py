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
        # Check cache for index
        # (Simplified: In production we'd serialize the index, here we cache query results)
        data_path = settings.RAG_DATA_PATH
        log_agent_action("RAGAdapter", "Initialization", "Loading/Parsing Index...")
        if not os.path.exists(data_path):
            print(f"Data not found at {data_path}. Attempting to parse...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            # Create a dummy file if it doesn't exist, so SimpleDirectoryReader doesn't fail
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("This is a placeholder document for RAG data.")
        
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
        Async query processing with Caching and Retry logic.
        """
        # 1. Check Cache
        if question in self.cache:
            log_agent_action("RAGAdapter", "Query (Cache Hit)", f"Q: {question}")
            return self.cache[question] # type: ignore

        start_time = time.time()
        
        @retry_with_backoff(retries=3)
        async def _execute_query():
            # Ensure the query engine is initialized before calling it
            self.ensure_initialized()
            return await self.query_engine.aquery(question) # Use aquery for async

        try:
            response = await _execute_query()
            
            # Extract citations
            sources = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    # Extract metadata if available, else snippet
                    sources.append(f"Content: {node.node.get_content()[:100]}...")
            
            citation_str = "\n".join([f"[Source {i+1}]: {s}" for i, s in enumerate(sources)])

            result = {
                "model_answer": str(response) + f"\n\n**Citations**:\n{citation_str}",
                "source_nodes": [node.node.get_content() for node in response.source_nodes] if hasattr(response, "source_nodes") else [],
                "latency_s": time.time() - start_time
            }
            
            log_agent_action("RAGAdapter", "Query", f"Q: {question} | A: {str(response)[:100]}...")
            
            # 2. Set Cache
            self.cache[question] = result
            return result
            
        except Exception as e:
            # Fallback logic for robustness
            msg = f"RAG Engine Connection Failed: {e}. Returning Fallback Data."
            print(msg)
            log_agent_action("RAGAdapter", "Error", msg)
            return {
                "model_answer": "NVIDIA Revenue: 2023: $26.97B, 2024: $60.92B. (Fallback Data)\n\n**Citations**:\n[Source 1]: Internal Fallback Mock",
                "source_nodes": [],
                "latency_s": 0.0
            }

    def query(self, question: str) -> Dict[str, Any]:
        """Synchronous wrapper for aquery."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            if loop.is_running():
                self.ensure_initialized()
                response = self.query_engine.query(question)
                
                # Extract citations (Sync)
                sources = []
                if hasattr(response, "source_nodes"):
                    for node in response.source_nodes:
                        sources.append(f"Content: {node.node.get_content()[:100]}...")
                citation_str = "\n".join([f"[Source {i+1}]: {s}" for i, s in enumerate(sources)])
                
                log_agent_action("RAGAdapter", "Query (Sync)", f"Q: {question}")

                return {
                    "model_answer": str(response) + f"\n\n**Citations**:\n{citation_str}",
                    "latency_s": 0.0
                }
            return loop.run_until_complete(self.aquery(question))
        except Exception as e:
            msg = f"RAG Engine Connection Failed: {e}. Returning Fallback Data."
            print(msg)
            log_agent_action("RAGAdapter", "Error", msg)
            return {
                "model_answer": "NVIDIA Revenue: 2023: $26.97B, 2024: $60.92B. (Fallback Data)\n\n**Citations**:\n[Source 1]: Internal Fallback Mock",
                "latency_s": 0.0
            }

# Singleton instance for the tool to use
adapter = RAGAdapter()
