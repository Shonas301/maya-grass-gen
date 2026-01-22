.PHONY: install install-dev lint typecheck test all clean

install:
	pip install -r requirements.in
	pip install -e .

install-dev:
	pip install -r requirements.in
	pip install -r requirements-dev.in
	pip install -e .

lint:
	ruff check .

typecheck:
	mypy src

test:
	pytest

all: lint typecheck test

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
