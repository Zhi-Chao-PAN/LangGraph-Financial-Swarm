# Contributing

Thanks for your interest in contributing to LangGraph Financial Swarm!

## Development Setup

```bash
git clone https://github.com/panzc/LangGraph-Financial-Swarm.git
cd LangGraph-Financial-Swarm
pip install -e ".[dev,pdf,web]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Code Quality

We use `ruff` for linting and formatting:

```bash
ruff check src/ tests/ eval/
ruff format src/ tests/ eval/
```

All code must pass lint and format checks before merging.

## Project Structure

- `src/agents/` — Supervisor, Researcher, Quant agents
- `src/core/` — Config, LLM adapters, conversation, templates, batch
- `src/data/` — PDF parser, data ingestion
- `src/tools/` — Plot tool
- `src/ui/` — Rich CLI, Streamlit Web UI, data explorer
- `src/utils/` — Robustness, validation, quality, tracker, observability
- `tests/` — Unit and integration tests
- `eval/` — Benchmark, ablation, latency analysis

## Adding a New Feature

1. Create your feature branch: `git checkout -b feat/my-feature`
2. Write tests first (TDD encouraged)
3. Implement the feature
4. Run `ruff check` and `ruff format`
5. Run `python -m pytest tests/` — all tests must pass
6. Commit with a descriptive message: `feat: description`

## Commit Convention

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Tests
- `chore:` — Maintenance
- `ci:` — CI/CD changes
- `release:` — Version release
