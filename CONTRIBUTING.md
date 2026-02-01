# Contributing to Financial Swarm

We adhere to the **Zero-Error Standard**.

## Development Setup
1. Clone the repo.
2. `pip install -r requirements.txt` (or use `Makefile`).
3. Ensure Ollama is running `deepseek-r1:8b`.

## Pull Request Process
1. **Test**: Run `pytest tests/` before committing.
2. **Lint**: Ensure no linting errors (we use `ruff`).
3. **Trace**: Verify your changes using `main.py --query "test"`.

## Style Guide
- **Docstrings**: Google Style.
- **Typing**: Strict type hints required.
- **Commits**: Conventional Commits (e.g., `feat: add new tool`).
