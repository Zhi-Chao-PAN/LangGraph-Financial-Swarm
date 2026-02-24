# LangGraph Financial Swarm

A hierarchical multi-agent system for financial analysis, built with [LangGraph](https://github.com/langchain-ai/langgraph) and locally deployed LLMs via [Ollama](https://ollama.ai).

## Overview

Traditional single-agent RAG pipelines struggle with multi-step financial reasoning — they fragment tables, lose context across pages, and hallucinate numbers. This project addresses these issues with a **Supervisor-Worker** architecture where specialized agents collaborate through a cyclic graph.

### Architecture

```
User Query
    |
    v
+--------------+
|  Supervisor   | <-- LLM Routing + Keyword Fallback
+------+-------+
       | delegates
       +------------------+
       v                  v
+--------------+   +--------------+
|  Researcher   |   |    Quant     |
| Query Rewrite |   | Visualization|
| + Multi-RAG   |   | bar/line/pie |
+------+-------+   +------+-------+
       |                  |
       +---> Supervisor <--+
                |
                v
             FINISH
```

- **Supervisor**: LLM-based intent routing with keyword fallback.
- **Researcher**: Query rewriting for complex questions, multi-query RAG with result synthesis.
- **Quant**: Generates bar, line, pie, stacked bar, and grouped bar charts.

### Key Features

- **Local-first**: All inference via Ollama. No data leaves your environment.
- **Structure-Aware RAG**: Documents chunked by section/table, each table is an independent retrieval unit.
- **Query Rewriting**: Complex questions decomposed into sub-queries, retrieved independently, then synthesized.
- **Multi-Company**: Financial data for NVIDIA, Apple, and Microsoft (FY2021-FY2025).
- **Robust Evaluation**: 40-question benchmark with approximate number matching (5% tolerance), entity matching, and partial scoring.

## Evaluation Results

| Mode | Accuracy | Partial Score | Hallucination |
|:---|---:|---:|---:|
| Swarm (Multi-Agent RAG) | 100.0% (30/30) | 0.935 | 0.0% |
| Baseline (Direct LLM) | 20.0% | 0.142 | 63.3% |

By difficulty: Easy 13/13, Medium 8/8, Hard 9/9.
By type: single_hop 13/13, comparison 8/8, calculation 6/6, multi_hop 3/3.

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) with a model (`ollama pull deepseek-r1:8b`)

### Installation

```bash
git clone https://github.com/Zhi-Chao-PAN/LangGraph-Financial-Swarm.git
cd LangGraph-Financial-Swarm
pip install .
```

### Configuration

Copy `.env.example` to `.env`:

```env
LLM_MODEL=deepseek-r1:8b
LLM_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

### Run

```bash
python main.py --query "What was NVIDIA's revenue in FY2025?"
python main.py --query "Compare NVIDIA and Apple gross margin in FY2025"
```

### Evaluate

```bash
python eval/run_eval.py --mode both
```

## Project Structure

```
├── main.py                  # Entry point (single query + interactive mode)
├── src/
│   ├── agents/
│   │   ├── supervisor.py    # LLM routing + keyword fallback
│   │   ├── researcher.py    # Query rewriting + multi-RAG + retry
│   │   └── chart_gen.py     # Visualization (6 chart types)
│   ├── core/
│   │   ├── config.py        # Pydantic settings
│   │   ├── llm.py           # DirectOllamaChat (LangChain)
│   │   ├── llm_index.py     # DirectOllamaLLM (LlamaIndex)
│   │   └── types.py         # State definitions
│   ├── tools/
│   │   └── plot_tool.py     # matplotlib chart generation
│   ├── utils/
│   │   ├── robustness.py    # Retry, logging, backoff
│   │   ├── validation.py    # Input sanitization, DataFrame validation
│   │   ├── observability.py # Rich UI tracing
│   │   └── parsing.py       # Robust JSON parsing
│   ├── rag_adapter.py       # Structure-aware RAG (section/table chunking)
│   └── data/ingest.py       # Document ingestion (PDF + Markdown + batch)
├── data/parsed/             # NVIDIA + Apple + Microsoft financials
├── eval/                    # 40-question benchmark + evaluation script
├── tests/                   # 52 tests
└── docs/                    # Architecture Decision Records
```

## Known Issues

- **Ollama 0.15.x compatibility**: The `ollama` Python library 0.6.1 returns 502 errors with Ollama server 0.15.2. This project uses `DirectOllamaChat` and `DirectOllamaLLM` wrappers that call the REST API directly via `requests`. Upgrading Ollama to 0.17.0+ should resolve this.

## Testing

```bash
pip install .[dev]
python -m pytest tests/ -v
```

52 tests covering: config, supervisor routing, keyword routing, input sanitization, number extraction, approximate matching, answer checking, plot generation (bar/line/pie/stacked/grouped), and query rewriting.

## License

MIT
