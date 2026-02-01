# src/rag_adapter.py
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
    Adapter for LlamaIndex RAG with persistent caching and non-blocking initialization.
    """
    def __init__(self) -> None:
        self.cache = dc.Cache(settings.DATA_DIR / "rag_cache")
        # 核心修改1: 初始化时不加载模型，移除副作用
        self.index = None 
        self.query_engine = None
        self._lock = asyncio.Lock() # 防止并发初始化竞争

    def _initialize_sync(self):
        """同步的、重型的初始化逻辑 (将在线程池中运行)"""
        log_agent_action("RAGAdapter", "Initialization", "Configuring Models & Loading Index...")
        
        # 核心修改2: 配置模型移到这里 (Lazy Config)
        Settings.embed_model = HuggingFaceEmbedding(model_name=settings.EMBEDDING_MODEL)
        Settings.llm = Ollama(model=settings.LLM_MODEL, base_url=settings.LLM_BASE_URL)
        
        # 加载数据
        data_path = settings.RAG_DATA_PATH
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"RAG data not found at {data_path}")
        
        documents = SimpleDirectoryReader(input_files=[data_path]).load_data()
        return VectorStoreIndex.from_documents(documents)

    async def _ensure_initialized_async(self):
        """异步封装初始化逻辑"""
        if self.query_engine is not None:
            return

        async with self._lock: # 确保只有一个协程执行初始化
            if self.query_engine is not None: # Double-check locking
                return
            
            loop = asyncio.get_running_loop()
            # 核心修改3: 将重型初始化扔到线程池执行，彻底释放 Event Loop
            self.index = await loop.run_in_executor(None, self._initialize_sync)
            self.query_engine = self.index.as_query_engine(similarity_top_k=3)

    async def aquery(self, question: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        
        # 1. Cache Check (Non-blocking)
        cached_result = await loop.run_in_executor(None, lambda: self.cache.get(question))
        if cached_result:
            log_agent_action("RAGAdapter", "Query (Cache Hit)", f"Q: {question}")
            return cached_result

        # 2. Lazy Load (Non-blocking!)
        await self._ensure_initialized_async()

        start_time = time.time()

        # 3. Query
        @retry_with_backoff(retries=3)
        async def _execute_query():
            return await self.query_engine.aquery(question)

        try:
            response = await _execute_query()
            
            # Extract citations logic
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
            
            log_agent_action("RAGAdapter", "Query", f"Q: {question}")
            
            # Set Cache
            await loop.run_in_executor(None, lambda: self.cache.set(question, result))
            return result
            
        except Exception as e:
            msg = f"RAG Engine Connection Failed: {e}"
            log_agent_action("RAGAdapter", "Error", msg)
            return {"model_answer": f"Error: {msg}", "source_nodes": [], "latency_s": 0.0}

# Singleton instance
adapter = RAGAdapter()
