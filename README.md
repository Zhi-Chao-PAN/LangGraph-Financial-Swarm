# LangGraph Financial Swarm

A hierarchical multi-agent system for financial analysis, built with [LangGraph](https://github.com/langchain-ai/langgraph) and locally deployed LLMs via [Ollama](https://ollama.ai).

## Overview

Traditional single-agent RAG pipelines struggle with multi-step financial reasoning — they fragment tables, lose context across pages, and hallucinate numbers. This project addresses these issues with a **Supervisor-Worker** architecture where specialized agents collaborate through a cyclic graph.

### Architecture

```
User Query
    │
    ▼
┌──────────────┐
│  Supervisor   │ ◄── LLM Routing + Keyword Fallback
└──────┬───────┘
       │ delegates
       ├──────────────────┐
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  Researcher   │   │    Quant     │
│ Deterministic │   │ Visualization│
│ Multi-Query   │   │ bar/line/pie │
│ + Hybrid RAG  │   └──────────────┘
│ + Retrieval   │
│  Verification │
└──────────────┘
       │
       └───▶ Supervisor ──▶ FINISH
```

- **Supervisor**: LLM-based intent routing with keyword fallback, max_rounds guard.
- **Researcher**: Deterministic multi-entity query decomposition, hybrid RAG (BM25 + vector), retrieval verification, confidence scoring.
- **Quant**: Generates bar, line, pie, stacked bar, and grouped bar charts.

### Key Features

- **Local-first**: All inference via Ollama (DeepSeek-R1:8B). No data leaves your environment.
- **Hybrid Search (v0.8.0)**: BM25 lexical + dense vector retrieval fused with Reciprocal Rank Fusion (RRF).
- **Structure-Aware RAG**: Documents chunked by section/table, each table is an independent retrieval unit.
- **Deterministic Multi-Query**: Multi-entity questions decomposed without LLM calls, with coverage verification.
- **Retrieval Verification**: After answering, checks if all entities are covered; auto-fills gaps with targeted queries.
- **PDF Ingestion**: Built-in PDF parser (PyMuPDF) extracts text + tables to Markdown for RAG. No API key needed.
- **Multi-Turn Conversation**: Follow-up detection, context injection, and `/history` command (v0.7.0).
- **Rich CLI UI (v0.8.0)**: Interactive terminal with status dashboard, formatted answer panels, eval tables, export, and cache management.
- **Graceful Degradation**: Ollama health check on startup; fallback messages when LLM is unavailable.
- **Multi-Company**: Financial data for NVIDIA, Apple, and Microsoft (FY2021–FY2025).

## Evaluation Results

60-question benchmark across NVIDIA, Apple, and Microsoft (FY2021–FY2025):

| Mode | Accuracy | Avg Partial Score | Hallucination |
|:---|---:|---:|---:|
| Swarm (v0.6.0) | 100.0% (60/60) | 0.900 | 0.0% |
| Baseline (Direct LLM) | 20.0% | 0.142 | 63.3% |

By difficulty: Easy 18/18, Medium 17/17, Hard 25/25.
By type: single_hop 18/18, comparison 21/21, calculation 13/13, multi_hop 8/8.

### Ablation Study

| Component Removed | Accuracy | Partial Score | Hallucination | Latency |
|:---|---:|---:|---:|---:|
| Full System | 97.5% | 0.927 | 0.0% | 18.0s |
| w/o Query Rewriting | 100.0% | 0.932 | 0.0% | 11.8s |
| w/o Structured Chunking | 92.5% | 0.848 | 5.0% | 80.7s |
| w/o Multi-Query Synthesis | 87.5% | 0.839 | 2.5% | 13.9s |

Key findings:
- Multi-Query Synthesis is the most critical component (−10% accuracy without it)
- Structured Chunking prevents hallucinations and reduces latency 4.5×
- LLM-based Query Rewriting was replaced with deterministic decomposition in v0.6.0

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with a model (`ollama pull deepseek-r1:8b`)

### Installation

```bash
git clone https://github.com/panzc/LangGraph-Financial-Swarm.git
cd LangGraph-Financial-Swarm
pip install -e ".[dev,pdf]"
```

### Interactive CLI

```bash
python -m src
```

This launches the Rich CLI with:
- System status dashboard (Ollama health, model info, data status)
- Interactive query mode with formatted answer panels
- Commands: `/ask`, `/chart`, `/history`, `/stats`, `/cache`, `/export`, `/status`

### Data Ingestion

```bash
# Ingest from PDF (local PyMuPDF, no API key needed)
python -m src.data.ingest --input data/raw/NVIDIA_10K.pdf --output data/parsed/nvidia.md

# Ingest from directory (PDF + Markdown)
python -m src.data.ingest --input data/raw/ --output data/parsed/combined.md
```

### Run Evaluation

```bash
python eval/run_eval.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
LangGraph-Financial-Swarm/
├── src/
│   ├── agents/          # Supervisor, Researcher, Quant agents
│   ├── core/            # Config, LLM adapters, conversation, types
│   ├── data/            # PDF parser, data ingestion pipeline
│   ├── tools/           # Plot tool for chart generation
│   ├── ui/              # Rich CLI (dashboard, display, interactive)
│   ├── utils/           # Robustness, validation, observability
│   └── rag_adapter.py   # Hybrid search (BM25 + vector RRF)
├── eval/                # Benchmark, ablation, latency analysis
├── tests/               # 215 tests (unit + integration)
├── data/                # Financial data (raw + parsed)
├── .github/workflows/   # CI/CD (lint + test)
└── pyproject.toml       # v0.9.0
```

## Version History

| Version | Highlights |
|:---|:---|
| v0.9.0 | Query templates, batch executor, answer quality assessment, trend detection, observability tracing, performance tracker, 215 tests |
| v0.8.0 | Hybrid search (BM25+RRF), Rich CLI UI, E2E PDF pipeline, health check, graceful degradation, CI/CD, Streamlit Web UI, Docker |
| v0.7.0 | Multi-turn conversation, follow-up detection, confidence scoring, latency analysis |
| v0.6.0 | Deterministic multi-query, retrieval verification, 60-question benchmark, PDF parser |
| v0.5.0 | Ablation study (4 experiments), structured chunking validation |
| v0.4.0 | Multi-company support (NVIDIA + Apple + Microsoft), 40-question benchmark |
| v0.3.0 | Structure-aware RAG, DirectOllamaChat/LLM bypass |
| v0.2.0 | Basic supervisor-worker graph, NVIDIA-only |

## Environment

- Hardware: ASUS Laptop, Intel Ultra 9 185H + RTX 4060 Laptop (8GB VRAM) + 40GB RAM
- OS: Windows 11
- Ollama: 0.15.2
- Model: deepseek-r1:8b
- Python: 3.12
- Embedding: BAAI/bge-large-en-v1.5 (CPU)

## License

MIT
