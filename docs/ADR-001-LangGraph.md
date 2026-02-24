# ADR 001: Selection of LangGraph for Hierarchical Agent Swarm

## Context
Financial document analysis requires coordinating multiple distinct capabilities:
1. **Precise Information Retrieval** (Structure-aware RAG from financial documents)
2. **Data Analysis & Visualization** (Chart generation with matplotlib)
3. **Strategic Oversight** (Query routing, decomposition, and result synthesis)

A flat multi-agent system often leads to circular discussions or loss of focus. We needed a framework that supports **Directed Cyclic Graphs (DCGs)** to enforce business logic.

## Decision
We selected **LangGraph** (built on LangChain) as the orchestration engine.

## Architecture (v0.3.0)

```
User Query → Supervisor (LLM routing + keyword fallback)
  → Researcher (query rewriting + multi-query RAG + synthesis)
  → Quant (6 chart types: bar, line, pie, stacked, grouped, scatter)
  → Supervisor → FINISH
```

Key design choices:
- **LLM routing with fallback**: Supervisor tries LLM-based intent detection first, falls back to keyword matching, then defaults to Researcher
- **Query rewriting**: Complex comparison questions are decomposed into sub-queries for independent retrieval
- **Structure-aware chunking**: Documents split by section/table boundaries, not arbitrary token windows
- **DirectOllama wrappers**: Bypass ollama Python library 502 bug via raw REST API calls

## Consequences

### Positive
- **State Control**: `AgentState` TypedDict provides typed, immutable audit log
- **Cyclic Capability**: Feedback loops (Supervisor rejects → agent retries)
- **Extensibility**: New agents added by creating a node + edge
- **Local-first**: All inference via Ollama, no data leaves the environment

### Negative
- **Complexity**: Higher learning curve than simple chains
- **Latency**: Multi-agent overhead (~28s per query with deepseek-r1:8b)

## Results
- 100% accuracy on 30-question financial QA benchmark
- 0% hallucination rate (vs 63.3% for direct LLM baseline)
