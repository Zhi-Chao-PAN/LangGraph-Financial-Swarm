# Security Policy

## Local-First Architecture

This project runs entirely locally via Ollama. No data is sent to external APIs by default.

## Reporting Vulnerabilities

If you discover a security issue, please email the maintainer directly rather than opening a public issue.

## Data Handling

- Financial data in `data/parsed/` is sourced from public SEC EDGAR filings
- No proprietary or confidential data is included
- RAG cache (`data/rag_cache/`) stores embeddings locally — clear it if sharing the repo

## Dependencies

- Ollama: Local LLM inference
- LlamaIndex: RAG framework (local embeddings via HuggingFace)
- LangGraph/LangChain: Agent orchestration
- matplotlib: Chart generation (no network access)

## API Keys

- `LLAMA_CLOUD_API_KEY`: Optional, only needed for LlamaParse PDF ingestion
- Store in `.env` (gitignored), never commit
