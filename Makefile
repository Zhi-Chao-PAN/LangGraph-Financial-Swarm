.PHONY: install test lint eval check clean

install:
	pip install .

install-dev:
	pip install .[dev]

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

eval:
	python eval/run_eval.py --mode both

eval-swarm:
	python eval/run_eval.py --mode swarm

eval-baseline:
	python eval/run_eval.py --mode baseline

check:
	python scripts/check_env.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache output/ data/rag_cache/
