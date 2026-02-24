# Contributing

## Setup

```bash
git clone https://github.com/Zhi-Chao-PAN/LangGraph-Financial-Swarm.git
cd LangGraph-Financial-Swarm
pip install .[dev]
```

## Development

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run tests: `python -m pytest tests/ -v`
4. Commit with clear messages
5. Open a pull request

## Code Style

- Python 3.10+ type hints
- Docstrings for public functions
- Keep agent nodes stateless (all state via `AgentState`)

## Adding a New Agent

1. Create `src/agents/your_agent.py` with a `create_your_agent_node(llm)` factory
2. Register in `main.py`: add node + edges
3. Update Supervisor's `members` list
4. Add tests in `tests/test_integration.py`

## Adding Financial Data

1. Add markdown tables to `data/parsed/llamaparse/parsed.md`
2. Use `## Section Title` headers (RAG chunks by section)
3. Add benchmark questions to `eval/benchmark.json`
4. Clear RAG cache: delete `data/rag_cache/`
