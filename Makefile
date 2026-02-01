.PHONY: setup test run check lint format docker-build clean

setup:
	pip install -r requirements.txt
	python check_env.py
	pre-commit install

test:
	pytest tests/

run:
	python main.py

check:
	python check_env.py

lint:
	ruff check .
	mypy .

format:
	ruff check . --fix

docker-build:
	docker build -t financial-swarm .

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
