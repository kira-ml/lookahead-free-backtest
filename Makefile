.PHONY: help install test coverage audit format lint clean run-pipeline run-all

help:
	@echo "Lookahead-Free Backtest Framework - Available Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo ""
	@echo "Testing & Validation:"
	@echo "  make test           Run all tests"
	@echo "  make coverage       Run tests with coverage report"
	@echo "  make audit          Run causality audit"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         Format code with black"
	@echo "  make lint           Check code style (future: add flake8/pylint)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make run-pipeline   Run full data pipeline (ingest -> features -> audit)"
	@echo "  make run-all        Run everything (install -> pipeline -> test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove generated files and caches"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

test:
	@echo "Running test suite..."
	pytest tests/ -v
	@echo "✓ Tests passed"

coverage:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

audit:
	@echo "Running causality audit..."
	python scripts/run_audit.py
	@echo "✓ Audit complete"

format:
	@echo "Formatting code with black..."
	black src/ tests/ scripts/
	@echo "✓ Code formatted"

lint:
	@echo "Linting code (placeholder for future linting setup)..."
	@echo "  To add: pip install flake8 pylint"
	@echo "  Then: flake8 src/ tests/"
	@echo "✓ Lint check complete"

clean:
	@echo "Cleaning up generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "✓ Cleanup complete"

run-pipeline:
	@echo "Running data pipeline..."
	@echo ""
	@echo "[1/3] Ingesting data..."
	python scripts/ingest_data.py
	@echo ""
	@echo "[2/3] Computing features..."
	python scripts/compute_features.py
	@echo ""
	@echo "[3/3] Running audit..."
	python scripts/run_audit.py
	@echo ""
	@echo "✓ Pipeline complete"

run-all: install run-pipeline test
	@echo ""
	@echo "============================================"
	@echo "✓ Full workflow complete!"
	@echo "============================================"
