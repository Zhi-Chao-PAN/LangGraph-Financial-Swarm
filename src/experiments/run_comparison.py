import os
import time
from typing import Dict, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Configure Global Settings as per user requirements
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
Settings.llm = Ollama(model="deepseek-r1:8b", request_timeout=60.0)

def load_index(filepath: str, name: str, device: str = "cpu") -> VectorStoreIndex:
    """
    输入: 
        filepath (str) - 解析后的 Markdown 文件路径
        name (str) - 实验名称
        device (str) - 运行设备
    输出: 
        VectorStoreIndex (LlamaIndex 对象)
    """
    # Simple loader for demo
    if os.path.isdir(filepath):
        documents = SimpleDirectoryReader(filepath).load_data()
    else:
        # Load single file as directory
        temp_dir = "data/parsed/temp"
        os.makedirs(temp_dir, exist_ok=True)
        filename = os.path.basename(filepath)
        import shutil
        shutil.copy(filepath, os.path.join(temp_dir, filename))
        documents = SimpleDirectoryReader(temp_dir).load_data()
        
    index = VectorStoreIndex.from_documents(documents)
    return index

async def evaluate_single_question(query_engine, row: Dict[str, Any], pipeline_name: str, semaphore=None, safe_mode=False) -> Dict[str, Any]:
    """
    输入语义: 
        query_engine: LlamaIndex 的查询引擎实例
        row: 包含 question 的字典
    """
    start_time = time.time()
    # Handle async query
    if hasattr(query_engine, "aquery"):
        response = await query_engine.aquery(row['question'])
    else:
        response = query_engine.query(row['question'])
    
    latency = time.time() - start_time
    
    return {
        "id": row.get("id", "N/A"),
        "question": row['question'],
        "ground_truth": row.get("ground_truth", "N/A"),
        "pipeline": pipeline_name,
        "model_answer": str(response),
        "latency_s": latency
    }
