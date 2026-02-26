# LangGraph Financial Swarm

A hierarchical multi-agent system for financial analysis, built with [LangGraph](https://github.com/langchain-ai/langgraph) and locally deployed LLMs via [Ollama](https://ollama.ai).

## Overview

Traditional single-agent RAG pipelines struggle with multi-step financial reasoning — they fragment tables, lose context across pages, and hallucinate numbers. This project addresses these issues with a **Supervisor-Worker** architecture where specialized agents collaborate through a cyclic graph.

### Architecture

```
User Query (EN/ZH)
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

**Core Intelligence**
- **Local-first**: All inference via Ollama (DeepSeek-R1:8B). No data leaves your environment.
- **Hybrid Search**: BM25 lexical + dense vector retrieval fused with Reciprocal Rank Fusion (RRF).
- **Structure-Aware RAG**: Documents chunked by section/table, each table is an independent retrieval unit.
- **Deterministic Multi-Query**: Multi-entity questions decomposed without LLM calls, with coverage verification.
- **Concurrent Sub-Queries**: Multi-entity queries execute in parallel via ThreadPoolExecutor.
- **Multi-LLM Backend (v1.0.0)**: Ollama / OpenAI / DeepSeek / Custom endpoints with auto-fallback.
- **Confidence Scoring (v1.0.0)**: Multi-signal answer confidence (hedging detection, source relevance, numeric evidence).
- **Source Attribution (v1.0.0)**: Trace which documents contributed to each answer.

**Real-Time Data**
- **Yahoo Finance Integration (v1.0.0)**: Fetch real-time data for any global public company.
- **Dynamic Index Hot-Update**: Add new company data to RAG index without restart.
- **Multi-Data-Source Plugin Architecture**: `DataSource` ABC + `SourceRegistry` for extensibility.
- **SQLite Cache**: Intelligent caching to minimize API calls.

**Financial Analysis**
- **Financial Calculator (v1.0.0)**: 17 ratios (profitability / liquidity / leverage / cash flow).
- **Peer Comparison (v1.0.0)**: Compare companies across 10 industry groups.
- **Trend Forecasting (v1.0.0)**: Linear regression + Z-score anomaly detection + moving average.
- **Report Generator (v1.0.0)**: Chinese/English bilingual reports with risk detection and health scoring.
- **Portfolio Tracker (v1.0.0)**: Track holdings, calculate returns, allocation breakdown.

**User Experience**
- **Chinese Query Support (v1.0.0)**: 80+ financial term mappings, language detection, response localization.
- **Smart Search Suggestions (v1.0.0)**: Trie prefix tree + hot query tracking + cold-start defaults.
- **Async Streaming (v1.0.0)**: SSE streaming responses for real-time answer generation.
- **Agent Reasoning Trace (v1.0.0)**: Mermaid flowchart visualization of multi-agent decision process.
- **Rich CLI UI**: Interactive terminal with dashboard, answer panels, eval tables, export.
- **Streamlit Web UI**: Browser-based interface with query, data explorer, and templates tabs.

**API & Deployment**
- **FastAPI REST API (v1.0.0)**: 10 endpoints — query, stream, fetch, report, compare, portfolio, backends, health.
- **pip Installable**: `pip install -e ".[all]"` with CLI entry point `financial-swarm`.
- **Docker + docker-compose**: One-command deployment.
- **GitHub Actions CI/CD**: Automated lint + test on every push.

## Evaluation Results

90-question benchmark across NVIDIA, Apple, Microsoft, Alphabet, and Amazon (FY2021–FY2025):

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

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with a model (`ollama pull deepseek-r1:8b`)

### Installation

```bash
git clone https://github.com/panzc/LangGraph-Financial-Swarm.git
cd LangGraph-Financial-Swarm
pip install -e ".[all,dev]"
```

### Interactive CLI

```bash
financial-swarm
# or
python -m src
```

### API Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

API endpoints:
- `POST /query` — Ask a financial question
- `POST /query/stream` — Stream answer via SSE
- `POST /fetch` — Fetch real-time data for any ticker
- `POST /report` — Generate financial health report (EN/ZH)
- `POST /compare` — Compare multiple companies
- `GET /portfolio` — View portfolio summary
- `POST /portfolio/add` — Add holding
- `POST /portfolio/remove` — Remove holding
- `GET /backends` — List LLM backends
- `GET /health` — System health check

### Run Tests

```bash
python -m pytest tests/ -v  # 513 tests
```

## Project Structure

```
LangGraph-Financial-Swarm/
├── src/
│   ├── agents/          # Supervisor, Researcher, Quant agents
│   ├── api/             # FastAPI REST API server
│   ├── core/            # Config, LLM backends, streaming, portfolio, suggestions
│   ├── data/            # PDF parser, Yahoo Finance fetcher, data sources
│   ├── tools/           # Plot tool for chart generation
│   ├── ui/              # Rich CLI + Streamlit Web UI
│   ├── utils/           # Calculator, report, trend, i18n, confidence, trace
│   └── rag_adapter.py   # Hybrid search (BM25 + vector RRF)
├── eval/                # Benchmark (90 questions), ablation
├── tests/               # 513 tests
├── data/                # Financial data (raw + parsed + cache)
├── .github/workflows/   # CI/CD
└── pyproject.toml       # v1.0.0
```

## Version History

| Version | Tests | Highlights |
|:---|---:|:---|
| v1.0.0 | 513 | Multi-LLM backend, FastAPI (10 endpoints), Yahoo Finance real-time, financial calculator (17 ratios), peer comparison, trend forecasting, report generator (CN/EN), portfolio tracker, i18n (80+ terms), smart suggestions, SSE streaming, agent reasoning trace, confidence scoring, source attribution, pip installable |
| v0.9.0 | 227 | Query templates, batch executor, answer quality, trend detection, concurrent queries, index persistence, Streamlit Web UI, observability, Docker |
| v0.8.0 | 142 | Hybrid search (BM25+RRF), Rich CLI UI, E2E PDF pipeline, health check, CI/CD |
| v0.7.0 | 94 | Multi-turn conversation, follow-up detection, confidence scoring |
| v0.6.0 | 60 | Deterministic multi-query, retrieval verification, 60-question benchmark |

## Environment

- Hardware: ASUS Laptop, Intel Ultra 9 185H + RTX 4060 Laptop (8GB VRAM) + 40GB RAM
- OS: Windows 11
- Model: deepseek-r1:8b via Ollama
- Python: 3.12
- Embedding: BAAI/bge-large-en-v1.5 (CPU)

## License

MIT
