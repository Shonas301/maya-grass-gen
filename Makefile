.PHONY: install install-dev lint typecheck test test-maya all clean

# path to mayapy - override with: make test-maya MAYAPY=/path/to/mayapy
MAYAPY ?= /System/Volumes/Data/Applications/Autodesk/maya2026/Maya.app/Contents/bin/mayapy

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
	pytest tests/unit

test-maya:
	$(MAYAPY) -m pytest tests/integration -v

all: lint typecheck test

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
