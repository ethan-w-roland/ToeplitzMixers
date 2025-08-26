.PHONY: lint lint-check

# Directories to exclude from linting
EXCLUDE_DIRS = .venv

# Dynamically discover experiment sub-directories, excluding specified dirs
EXPERIMENT_DIRS := $(filter-out $(EXCLUDE_DIRS), $(shell find experiments -maxdepth 1 -mindepth 1 -type d))

lint:
	@for dir in $(EXPERIMENT_DIRS); do \
		echo "Running linters in $$dir"; \
		black $$dir; \
		nbqa black $$dir; \
		ruff check --fix $$dir; \
		nbqa ruff --fix $$dir; \
		find $$dir -name "*.py" -o -name "*.pyi" | grep -q . && mypy $$dir --ignore-missing-imports --no-namespace-packages || true; \
		nbqa mypy $$dir --ignore-missing-imports; \
	done

lint-check:
	@for dir in $(EXPERIMENT_DIRS); do \
		echo "Running lint check in $$dir"; \
		black $$dir --check; \
		nbqa black $$dir --check; \
		ruff check $$dir; \
		nbqa ruff $$dir; \
		find $$dir -name "*.py" -o -name "*.pyi" | grep -q . && mypy $$dir --ignore-missing-imports --no-namespace-packages || true; \
		nbqa mypy $$dir --ignore-missing-imports; \
	done