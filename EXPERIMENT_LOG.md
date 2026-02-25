# Experiment Log — LangGraph Financial Swarm

## Ablation Study (Phase 4)

| Experiment | Date | Model | Params | Accuracy | Partial Score | Hallucination | Avg Latency | GPU | Notes |
|---|---|---|---|---|---|---|---|---|---|
| full | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, query rewrite, multi-query | 97.5% (39/40) | 0.927 | 0% | 18.0s | RTX 4060 Laptop | Q34 missed (3-company comparison, synthesis lost Apple+MSFT data) |
| no_rewrite | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, NO query rewrite, multi-query | 100% (40/40) | 0.932 | 0% | 11.8s | RTX 4060 Laptop | Better than full — rewrite adds noise + latency |
| no_multiquery | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, query rewrite, NO multi-query | 87.5% (35/40) | 0.839 | 2.5% | 13.9s | RTX 4060 Laptop | Biggest drop — comparison type 8/13, 1 hallucination |
| no_structured | 2025-02-25 | deepseek-r1:8b | top_k=5, SentenceSplitter chunking, query rewrite, multi-query | 92.5% (37/40) | 0.848 | 5.0% | 80.7s | RTX 4060 Laptop | Naive chunking: single_hop 16/18, hallucination 5%, latency 4.5x |

## Phase 3 Results (Baseline)

| Experiment | Date | Model | Accuracy | Partial Score | Hallucination | Notes |
|---|---|---|---|---|---|---|
| Swarm (3 companies) | 2025-02-25 | deepseek-r1:8b | 100% (40/40) | 0.935 | 0% | NVIDIA + Apple + Microsoft, 40 questions |
| Baseline (direct LLM) | 2025-02-25 | deepseek-r1:8b | 20.0% | 0.142 | 63.3% | No RAG, direct LLM answer |

## Phase 2 Results

| Experiment | Date | Model | Accuracy | Partial Score | Hallucination | Notes |
|---|---|---|---|---|---|---|
| Swarm (NVIDIA only) | 2025-02-24 | deepseek-r1:8b | 100% (30/30) | 0.935 | 0% | 30 questions, NVIDIA only |
| Baseline | 2025-02-24 | deepseek-r1:8b | 20.0% | 0.142 | 63.3% | |

## Key Findings

1. **Multi-Query Synthesis is critical** — removing it drops accuracy by 10% (97.5% → 87.5%), especially on comparison questions (12/13 → 8/13)
2. **Structured Chunking matters** — removing it drops accuracy to 92.5%, introduces 5% hallucination, and increases latency 4.5x (18s → 80.7s)
3. **Query Rewriting is harmful on this benchmark** — removing it improves accuracy (97.5% → 100%) and reduces latency (18s → 11.8s)
4. **single_hop and calculation are robust** — unaffected by multi-query/rewrite ablation, always 100%
5. **Naive chunking hurts even simple questions** — single_hop drops from 18/18 to 16/18 with SentenceSplitter

## Environment

- Hardware: ASUS Laptop, Intel Ultra 9 185H + RTX 4060 Laptop (8GB VRAM) + 40GB RAM
- OS: Windows 11
- Ollama: 0.15.2
- Model: deepseek-r1:8b
- Python: 3.12
- LangGraph: latest
- LlamaIndex: latest
- Embedding: BAAI/bge-large-en-v1.5 (CPU)
