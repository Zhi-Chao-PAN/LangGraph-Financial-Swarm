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
| Deterministic |   | Visualization|
| Multi-Query   |   | bar/line/pie |
| + Retrieval   |   +--------------+
|  Verification |
+--------------+
       |
       +---> Supervisor --> FINISH
```

- **Supervisor**: LLM-based intent routing with keyword fallback.
- **Researcher**: Deterministic multi-entity query decomposition, multi-query RAG with retrieval verification.
- **Quant**: Generates bar, line, pie, stacked bar, and grouped bar charts.

### Key Features

- **Local-first**: All inference via Ollama (DeepSeek-R1:8B). No data leaves your environment.
- **Structure-Aware RAG**: Documents chunked by section/table, each table is an independent retrieval unit.
- **Deterministic Multi-Query**: Multi-entity questions decomposed without LLM calls, with coverage verification.
- **Retrieval Verification**: After answering, checks if all entities are covered; auto-fills gaps with targeted queries.
- **PDF Ingestion**: Built-in PDF parser (PyMuPDF) extracts text + tables to Markdown for RAG.
- **Multi-Company**: Financial data for NVIDIA, Apple, and Microsoft (FY2021-FY2025).

## Evaluation Results

60-question benchmark across NVIDIA, Apple, and Microsoft (FY2021-FY2025):

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
- Multi-Query Synthesis is the most critical component (-10% accuracy without it)
- Structured Chunking prevents hallucinations and reduces latency 4.5x
- LLM-based Query Rewriting was replaced with deterministic decomposition in v0.6.0

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
# Single query
python main.py --query "What was NVIDIA's revenue in FY2025?"

# Multi-company comparison
python main.py --query "Compare NVIDIA and Apple gross margin in FY2025"

# Parse PDF and ingest
python -m src.data.pdf_parser --input report.pdf --output data/parsed/output.md
```

### Evaluate

```bash
# Full evaluation (60 questions)
python eval/run_eval.py --mode both

# Ablation study
python eval/run_ablation.py --mode all
```

## Project Structure

```
├── main.py                  # Entry point
├── src/
│   ├── agents/
│   │   ├── supervisor.py    # LLM routing + keyword fallback
│   │   ├── researcher.py    # Deterministic multi-query + retrieval verification
│   │   └── chart_gen.py     # Visualization (6 chart types)
│   ├── core/
│   │   ├── config.py        # Pydantic settings
│   │   ├── llm.py           # DirectOllamaChat (bypasses ollama lib 502 bug)
│   │   ├── llm_index.py     # DirectOllamaLLM (LlamaIndex compatible)
│   │   └── types.py         # State definitions
│   ├── data/
│   │   └── pdf_parser.py    # PDF to Markdown (PyMuPDF)
│   ├── tools/
│   │   └── plot_tool.py     # Matplotlib chart generation
│   ├── utils/
│   │   ├── robustness.py    # Retry, logging, backoff
│   │   ├── validation.py    # Input sanitization
│   │   ├── observability.py # Rich UI tracing
│   │   └── parsing.py       # Robust JSON parsing
│   └── rag_adapter.py       # Structure-aware RAG (section/table chunking)
├── eval/
│   ├── benchmark.json       # 60-question benchmark
│   ├── run_eval.py          # Evaluation script
│   ├── run_ablation.py      # Ablation study (4 modes)
│   └── plot_ablation.py     # Publication-ready charts
├── data/parsed/             # NVIDIA + Apple + Microsoft financials
├── tests/                   # 52 tests
├── EXPERIMENT_LOG.md        # All experiment results with parameters
└── docs/                    # Architecture Decision Records
```

## Testing

```bash
pip install .[dev]
python -m pytest tests/ -v
```

52 tests covering: config, supervisor routing, input sanitization, number extraction, approximate matching, answer checking, plot generation, and query rewriting.

## License

MIT
