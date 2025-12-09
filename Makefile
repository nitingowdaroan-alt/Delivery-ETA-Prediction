# =============================================================================
# Delivery ETA Prediction - Makefile
# =============================================================================
# Commands for development, training, testing, and deployment
#
# Usage:
#   make install          # Install dependencies
#   make generate-data    # Generate synthetic training data
#   make train            # Train the model
#   make serve            # Run API server
#   make test             # Run tests
#   make docker-up        # Start all services with Docker

.PHONY: help install install-dev generate-data clean-data train train-optimized \
        evaluate serve serve-dev test test-unit test-integration test-load \
        docker-build docker-up docker-down docker-logs lint format clean

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
POETRY := poetry
DOCKER_COMPOSE := docker-compose
API_PORT := 8000

# Colors for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Delivery ETA Prediction System$(NC)"
	@echo "================================="
	@echo ""
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies with Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(POETRY) install --no-dev
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install all dependencies including dev tools
	@echo "$(BLUE)Installing all dependencies...$(NC)"
	$(POETRY) install
	@echo "$(GREEN)Installation complete!$(NC)"

install-pip: ## Install with pip (alternative to Poetry)
	@echo "$(BLUE)Installing with pip...$(NC)"
	pip install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

# =============================================================================
# Data Generation and Processing
# =============================================================================

generate-data: ## Generate synthetic training data (100k+ rows)
	@echo "$(BLUE)Generating synthetic delivery data...$(NC)"
	@mkdir -p data/raw data/processed logs
	$(POETRY) run python -m src.data.generate
	@echo "$(GREEN)Data generation complete! See data/raw/$(NC)"

clean-data: ## Clean and preprocess raw data
	@echo "$(BLUE)Cleaning data...$(NC)"
	$(POETRY) run python -c "from src.data.clean import clean_data; clean_data('data/raw/delivery_orders.csv', 'data/processed/delivery_orders_clean.csv')"
	@echo "$(GREEN)Data cleaning complete! See data/processed/$(NC)"

# =============================================================================
# Model Training
# =============================================================================

train: ## Train model with default parameters
	@echo "$(BLUE)Training delivery ETA model...$(NC)"
	@mkdir -p models logs
	$(POETRY) run python -m src.models.train --data data/raw/delivery_orders.csv
	@echo "$(GREEN)Training complete! Model saved to models/$(NC)"

train-optimized: ## Train with Optuna hyperparameter optimization
	@echo "$(BLUE)Training with hyperparameter optimization...$(NC)"
	@mkdir -p models logs
	$(POETRY) run python -m src.models.train \
		--data data/raw/delivery_orders.csv \
		--optimize \
		--n-trials 50 \
		--timeout 1800
	@echo "$(GREEN)Optimized training complete!$(NC)"

train-quick: ## Quick training (for testing pipeline)
	@echo "$(BLUE)Quick training run...$(NC)"
	@mkdir -p models logs
	$(POETRY) run python -m src.models.train \
		--data data/raw/delivery_orders.csv \
		--n-trials 5 \
		--timeout 300
	@echo "$(GREEN)Quick training complete!$(NC)"

evaluate: ## Evaluate trained model and generate reports
	@echo "$(BLUE)Evaluating model...$(NC)"
	@mkdir -p reports
	$(POETRY) run python -m src.models.evaluate
	@echo "$(GREEN)Evaluation complete! See reports/$(NC)"

# =============================================================================
# API Server
# =============================================================================

serve: ## Run production API server
	@echo "$(BLUE)Starting API server on port $(API_PORT)...$(NC)"
	$(POETRY) run uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT)

serve-dev: ## Run development server with auto-reload
	@echo "$(BLUE)Starting development server...$(NC)"
	$(POETRY) run uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

serve-prod: ## Run production server with multiple workers
	@echo "$(BLUE)Starting production server with 4 workers...$(NC)"
	$(POETRY) run uvicorn src.api.main:app --host 0.0.0.0 --port $(API_PORT) --workers 4

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(POETRY) run pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "$(GREEN)All tests complete!$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(POETRY) run pytest tests/test_features.py -v

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(POETRY) run pytest tests/test_api.py -v

test-load: ## Run load tests with Locust (requires running server)
	@echo "$(BLUE)Running load tests...$(NC)"
	@echo "$(YELLOW)Make sure the API server is running (make serve)$(NC)"
	$(POETRY) run locust -f tests/load_test.py --host=http://localhost:$(API_PORT) \
		-u 10 -r 2 --run-time 60s --headless --html=reports/load_test.html
	@echo "$(GREEN)Load test complete! See reports/load_test.html$(NC)"

test-load-heavy: ## Run heavy load test (100 users)
	@echo "$(BLUE)Running heavy load test...$(NC)"
	$(POETRY) run locust -f tests/load_test.py --host=http://localhost:$(API_PORT) \
		-u 100 -r 10 --run-time 120s --headless --html=reports/load_test_heavy.html

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build
	@echo "$(GREEN)Build complete!$(NC)"

docker-up: ## Start all services with Docker Compose
	@echo "$(BLUE)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "  API:        http://localhost:8000"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-logs: ## View Docker service logs
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Remove all Docker volumes and images
	@echo "$(RED)WARNING: This will remove all data!$(NC)"
	$(DOCKER_COMPOSE) down -v --rmi all

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting checks
	@echo "$(BLUE)Running linters...$(NC)"
	$(POETRY) run flake8 src/ tests/
	$(POETRY) run mypy src/

format: ## Format code with Black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(POETRY) run black src/ tests/
	$(POETRY) run isort src/ tests/
	@echo "$(GREEN)Formatting complete!$(NC)"

format-check: ## Check code formatting without changes
	@echo "$(BLUE)Checking formatting...$(NC)"
	$(POETRY) run black --check src/ tests/
	$(POETRY) run isort --check-only src/ tests/

# =============================================================================
# MLflow
# =============================================================================

mlflow-ui: ## Start MLflow UI
	@echo "$(BLUE)Starting MLflow UI...$(NC)"
	$(POETRY) run mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000

# =============================================================================
# Monitoring
# =============================================================================

monitor-drift: ## Run drift detection on recent data
	@echo "$(BLUE)Running drift detection...$(NC)"
	$(POETRY) run python -m src.monitoring.drift
	@echo "$(GREEN)Drift report generated! See reports/monitoring/$(NC)"

# =============================================================================
# Utilities
# =============================================================================

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage
	rm -rf src/__pycache__ tests/__pycache__
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Clean complete!$(NC)"

clean-all: clean ## Clean all generated files including data and models
	@echo "$(RED)WARNING: This will remove data and trained models!$(NC)"
	rm -rf data/raw/* data/processed/* models/*.joblib logs/*

dirs: ## Create necessary directories
	@mkdir -p data/raw data/processed models logs reports/monitoring config

# =============================================================================
# Quick Start (Full Pipeline)
# =============================================================================

quickstart: install-dev generate-data train serve-dev ## Complete setup: install, generate data, train, serve
	@echo "$(GREEN)Quick start complete! API running at http://localhost:$(API_PORT)$(NC)"

pipeline: generate-data clean-data train evaluate ## Run full ML pipeline
	@echo "$(GREEN)Full pipeline complete!$(NC)"
	@echo "  - Data generated and cleaned"
	@echo "  - Model trained"
	@echo "  - Evaluation reports in reports/"

# =============================================================================
# Sample Commands
# =============================================================================

sample-predict: ## Make a sample prediction request
	@echo "$(BLUE)Making sample prediction...$(NC)"
	@curl -s -X POST http://localhost:$(API_PORT)/predict/realtime \
		-H "Content-Type: application/json" \
		-d '{"pickup_lat": 37.7879, "pickup_lng": -122.4074, "dropoff_lat": 37.7749, "dropoff_lng": -122.4194, "zone_id": 1, "restaurant_type": "casual", "traffic_multiplier": 1.2}' | python -m json.tool

sample-health: ## Check API health
	@curl -s http://localhost:$(API_PORT)/health | python -m json.tool

sample-metrics: ## View Prometheus metrics
	@curl -s http://localhost:$(API_PORT)/metrics | head -50
