# Thin wrappers so a local run matches CI by construction.
# CI runs `pre-commit` and `pytest` with the same configuration.

.PHONY: install lint format test test-slow test-all build clean

install:
	pip install -e .[test,dev]
	pre-commit install

lint:
	pre-commit run --all-files

format:
	ruff format .
	ruff check --fix .

# Default selection, same as the CI `test` job: benchmark_sweep, slow and
# download tests are deselected (see [tool.pytest.ini_options] in pyproject.toml).
test:
	pytest

# Same as the weekly CI `slow` job.
test-slow:
	pytest -m "slow or download"

# Everything, including the dataset-download and benchmark tests.
test-all:
	pytest -m ""

build:
	python -m build
	python -m twine check --strict dist/*

clean:
	rm -rf dist build .pytest_cache .ruff_cache .benchmarks
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
