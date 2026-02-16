.PHONY: install install-dev lint lint-fix typecheck test test-maya package verify-package all clean release

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

lint-fix:
	ruff check --fix .

typecheck:
	mypy src

test:
	pytest tests/unit

test-maya:
	$(MAYAPY) -m pytest tests/integration -v

package:
	./scripts/create_zip.sh

verify-package:
	./scripts/verify_plugin.sh --zip dist/maya-grass-gen.zip

all: lint typecheck test

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# cut a release: make release VERSION=1.2.0
# add DRY_RUN=1 to preview without making changes
release:
ifndef VERSION
	$(error VERSION is required — usage: make release VERSION=1.2.0)
endif
	./scripts/release.sh $(if $(DRY_RUN),--dry-run) $(VERSION)
