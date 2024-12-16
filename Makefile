.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all        - Run all checks (lint, format, types)"
	@echo "  all-fix    - Run all checks with auto-fixing (lint-fix, format-fix, types)"
	@echo "  lint       - Run ruff linter"
	@echo "  lint-fix   - Run ruff linter with auto-fixing"
	@echo "  format     - Run ruff formatter"
	@echo "  format-fix - Run ruff formatter with auto-fixing"
	@echo "  types      - Run mypy type checker"

CONFIG ?= pyproject.toml
PWD := $(shell pwd)

# Optional mode to check status of files using formatting tools without formatting.
MODE ?=

all: lint format types

all-fix: lint-fix format-fix types

lint:
	@echo "==============================================================="
	@echo "Running Ruff linter in directory $(PWD) with config file: $(CONFIG)"
	@ruff check --config $(CONFIG) $(MODE) .
	@echo "Ruff linting complete."

lint-fix:
	@echo "==============================================================="
	@echo "Running Ruff linter with auto-fix in directory $(PWD) with config file: $(CONFIG)"
	@ruff check --fix --config $(CONFIG) $(MODE) .
	@echo "Ruff linting with fixes complete."

format:
	@echo "==============================================================="
	@echo "Running Ruff formatter in directory $(PWD) with config file: $(CONFIG)"
	@ruff format --check --config $(CONFIG) $(MODE) .
	@echo "Ruff formatting complete."

format-fix:
	@echo "==============================================================="
	@echo "Running Ruff formatter with auto-fix in directory $(PWD) with config file: $(CONFIG)"
	@ruff format --config $(CONFIG) $(MODE) .
	@echo "Ruff formatting with fixes complete."

types:
	@echo "==============================================================="
	@echo "Running mypy in directory $(PWD) with config file: $(CONFIG)"
	@mypy --config-file $(CONFIG) .
	@echo "mypy complete."