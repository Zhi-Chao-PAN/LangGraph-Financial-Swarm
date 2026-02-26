# Experiment Log — LangGraph Financial Swarm

## Ablation Study (Phase 4)

| Experiment | Date | Model | Params | Accuracy | Partial Score | Hallucination | Avg Latency | GPU | Notes |
|---|---|---|---|---|---|---|---|---|---|
| full | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, query rewrite, multi-query | 97.5% (39/40) | 0.927 | 0% | 18.0s | RTX 4060 Laptop | Q34 missed (3-company comparison, synthesis lost Apple+MSFT data) |
| no_rewrite | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, NO query rewrite, multi-query | 100% (40/40) | 0.932 | 0% | 11.8s | RTX 4060 Laptop | Better than full — rewrite adds noise + latency |
| no_multiquery | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, query rewrite, NO multi-query | 87.5% (35/40) | 0.839 | 2.5% | 13.9s | RTX 4060 Laptop | Biggest drop — comparison type 8/13, 1 hallucination |
| no_structured | 2025-02-25 | deepseek-r1:8b | top_k=5, SentenceSplitter chunking, query rewrite, multi-query | 92.5% (37/40) | 0.848 | 5.0% | 80.7s | RTX 4060 Laptop | Naive chunking: single_hop 16/18, hallucination 5%, latency 4.5x |
| v0.6.0 (improved) | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, deterministic multi-query, retrieval verification | 100% (40/40) | 0.942 | 0% | 32.8s | RTX 4060 Laptop | Replaced LLM rewrite with deterministic decomposition + coverage check |
| v0.6.0 (60Q) | 2025-02-25 | deepseek-r1:8b | top_k=5, structured chunking, deterministic multi-query, retrieval verification | 100% (60/60) | 0.900 | 0% | ~35s | RTX 4060 Laptop | Expanded benchmark: 20 new hard questions, all correct |

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

## Code Quality & Infrastructure (2025-02-26)

| Task | Status | Details |
|---|---|---|
| Ruff lint | 0 errors | 148 issues fixed (auto + manual), line-length 150 |
| mypy type check | 0 errors | config.py, robustness.py, validation.py typed |
| Tests | 81 passing | +11 PDF parser, +18 conversation memory |
| Pre-commit | Configured | ruff + format + trailing whitespace + large file check |
| Structured logging | Done | log_agent_action supports info/warning/error levels |
| Supervisor guard | Done | max_rounds=10 prevents infinite loops |
| Cache TTL | Done | RAG cache expires after 24h |
| Input validation | Done | Query truncated at 1000 chars |
| LaTeX tables | 3 generated | ablation_main.tex, ablation_difficulty.tex, ablation_qtype.tex |
| PDF E2E test | Passed | Apple 213K chars/199 tables, Microsoft 345K chars/6 tables |
| Conversation memory | v0.7.0 | Follow-up detection, context injection, ConversationManager |

## v0.8.0 Features (2025-02-26)

| Task | Status | Details |
|---|---|---|
| Hybrid Search | Done | BM25 + Vector retrieval, Reciprocal Rank Fusion (RRF), 20 tests |
| Rich CLI UI | Done | Banner, status dashboard, answer panels, eval tables, cache mgmt, export |
| E2E PDF Ingest | Done | Local PyMuPDF parser (no API key), directory batch ingest, 7 tests |
| Ollama Health Check | Done | Startup health check, graceful LLM degradation, 5 tests |
| CI/CD | Done | GitHub Actions: ruff lint + format check + pytest |
| Input Validation | Done | sanitize_input, sanitize_rag_query, validate_dataframe decorator |
| Code Quality | 0 errors | ruff lint + format, 142 tests passing |
| Version | v0.8.0 | pyproject.toml, README rewrite, __main__.py entry point |

## v0.9.0 Features (2025-02-26)

| Task | Status | Details |
|---|---|---|
| Query Templates | Done | 12 templates (single_hop/comparison/calculation/trend), batch generation |
| Batch Executor | Done | Sequential execution, progress callback, file/JSON loading, 14 tests |
| Answer Quality | Done | Empty/weak/hedging/repetition detection, 9 tests |
| Trend Detection | Done | FY range parsing, trend query decomposition, 8 tests |
| Performance Tracker | Done | QueryRecord, p95 latency, save/load JSON, 12 tests |
| Observability | Done | Query ID, per-agent latency, error tracking, final report table, 9 tests |
| Config Validation | Done | Startup warnings for bad settings, 4 tests |
| Streamlit Web UI | Done | Dashboard, query, history, export, cache mgmt |
| Docker | Done | Dockerfile + docker-compose with Ollama |
| Tests | 227 total | +85 from v0.8.0 |
| Concurrent Queries | Done | ThreadPoolExecutor for multi-entity, max 4 workers, 2 tests |
| Index Persistence | Done | MD5 hash check, auto-rebuild on data change, faster startup |

## Environment

- Hardware: ASUS Laptop, Intel Ultra 9 185H + RTX 4060 Laptop (8GB VRAM) + 40GB RAM
- OS: Windows 11
- Ollama: 0.15.2
- Model: deepseek-r1:8b
- Python: 3.12
- LangGraph: latest
- LlamaIndex: latest
- Embedding: BAAI/bge-large-en-v1.5 (CPU)
