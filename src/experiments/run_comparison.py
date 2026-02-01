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
    

import asyncio
import argparse
import json
from src.core.config import settings

import logging
# Configure logger for experiment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Experiment")

async def run_benchmark(output_file: str = "experiments/results.json"):
    logger.info("Initializing Experiment...")
    
    # 1. Setup RAG Engine (using Adapter logic or direct loading)
    # For comparison, we might want to compare "Standard RAG" vs "Structure-Aware"
    # Here we assume 'Standard' is just the same index but maybe queried differently, 
    # or we load a different index. For simplicity in this artifact, we run one pipeline.
    
    # Check if index exists
    index_path = settings.RAG_DATA_PATH
    if not os.path.exists(index_path):
        logger.warning(f"Index not found at {index_path}. Using placeholder.")
    
    # Load Index (Structure-Aware by default as per our setup)
    # In a real comparison, we'd load two indices.
    try:
        index = load_index(index_path, "Structure-Aware")
        query_engine = index.as_query_engine(similarity_top_k=3)
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        return

    # 2. Define Benchmark (Golden Set)
    # As per README: 50 Multi-hop financial questions.
    # We'll use a subset here for the runnable script if file doesn't exist.
    benchmark_file = "data/benchmark_n50.json"
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            questions = json.load(f)
    else:
        logger.info("Benchmark file not found. Using sample set.")
        questions = [
            {"id": 1, "question": "What was NVIDIA's revenue in 2023?", "ground_truth": "26.974"}, # Billion (normalized)
            {"id": 2, "question": "Compare AMD and NVIDIA 2024 gross margin.", "ground_truth": "NVIDIA: 72.7%, AMD: 46%"}, 
            {"id": 3, "question": "Did NVIDIA's R&D expenses increase in 2024?", "ground_truth": "Yes, to $8.68B"}
        ]

    results = []
    logger.info(f"Running evaluation on {len(questions)} queries...")
    
    for row in questions:
        logger.info(f"Processing Q{row['id']}: {row['question']}")
        try:
            res = await evaluate_single_question(query_engine, row, "Structure-Aware")
            results.append(res)
        except Exception as e:
            logger.error(f"Error on Q{row['id']}: {e}")
            
    # 3. Save Results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")
    
    # 4. Calculate Metrics
    try:
        from src.experiments.evaluate_metrics import calculate_metrics
        metrics = calculate_metrics(results)
        print("\n=== Auto-Evaluation Metrics ===") # Keep strict output for pipe-ability
        print(f"Exact Match Accuracy: {metrics['accuracy']:.2%}")
        print(f"Average Latency: {metrics['avg_latency']:.4f}s")
        print("===============================\n")
    except ImportError:
        logger.error("Could not import metric evaluator.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/comparison_results.json")
    args = parser.parse_args()
    
    # Run async loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    loop.run_until_complete(run_benchmark(args.output))

if __name__ == "__main__":
    main()
