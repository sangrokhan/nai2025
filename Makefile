.PHONY: help check test install clean format lint validate build

# Colors
GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
WHITE  := $(shell tput -Txterm setaf 7)
RESET  := $(shell tput -Txterm sgr0)

help: ## Show this help message
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} { \
		if (/^[a-zA-Z_-]+:.*?##.*$$/) {printf "  ${YELLOW}%-20s${GREEN}%s${RESET}\n", $$1, $$2} \
		else if (/^## .*$$/) {printf "  ${WHITE}%s${RESET}\n", substr($$1,4)} \
		}' $(MAKEFILE_LIST)

## Development

validate: ## Run all validation checks (no dependencies required)
	@echo "${GREEN}Running validation checks...${RESET}"
	python3 check_all.py

check: ## Quick syntax and structure check
	@echo "${GREEN}Running quick checks...${RESET}"
	python3 validate_code.py

install: ## Install package and dependencies
	@echo "${GREEN}Installing package...${RESET}"
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install package with development dependencies
	@echo "${GREEN}Installing package with dev dependencies...${RESET}"
	pip install -r requirements.txt
	pip install -e ".[dev]"

## Testing

test: ## Run all tests (requires dependencies)
	@echo "${GREEN}Running all tests...${RESET}"
	pytest tests/ -v

test-unit: ## Run only unit tests
	@echo "${GREEN}Running unit tests...${RESET}"
	pytest tests/test_models.py tests/test_data.py tests/test_training.py -v

test-integration: ## Run integration tests
	@echo "${GREEN}Running integration tests...${RESET}"
	pytest tests/test_integration.py -v

test-cov: ## Run tests with coverage report
	@echo "${GREEN}Running tests with coverage...${RESET}"
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
	@echo "${GREEN}Coverage report saved to htmlcov/index.html${RESET}"

## Code Quality

format: ## Format code with black and isort
	@echo "${GREEN}Formatting code...${RESET}"
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint: ## Lint code with flake8
	@echo "${GREEN}Linting code...${RESET}"
	flake8 src/ tests/ scripts/ --max-line-length=100 --extend-ignore=E203,W503

lint-strict: ## Strict linting (stops on errors)
	@echo "${GREEN}Running strict lint...${RESET}"
	flake8 src/ tests/ scripts/ --count --select=E9,F63,F7,F82 --show-source --statistics

## Training

train-small: ## Train small model
	@echo "${GREEN}Training small model...${RESET}"
	python scripts/train.py --config configs/small_model.yaml

train-medium: ## Train medium model (recommended)
	@echo "${GREEN}Training medium model...${RESET}"
	python scripts/train.py --config configs/medium_model.yaml

train-large: ## Train large model
	@echo "${GREEN}Training large model...${RESET}"
	python scripts/train.py --config configs/large_model.yaml

train-debug: ## Train with debug mode
	@echo "${GREEN}Training with debug mode...${RESET}"
	python scripts/train.py --config configs/medium_model.yaml --debug

## Data

generate-data: ## Generate sample test data
	@echo "${GREEN}Generating sample data...${RESET}"
	@python3 -c "import numpy" 2>/dev/null || (echo "${YELLOW}Installing data dependencies...${RESET}" && pip install numpy pandas pyarrow)
	python3 generate_sample_data.py

check-data: ## Check if data exists
	@echo "${GREEN}Checking data files...${RESET}"
	@if [ -f data/train.parquet ]; then \
		echo "  ✓ train.parquet exists"; \
	else \
		echo "  ✗ train.parquet missing (run 'make generate-data')"; \
	fi
	@if [ -f data/val.parquet ]; then \
		echo "  ✓ val.parquet exists"; \
	else \
		echo "  ✗ val.parquet missing (run 'make generate-data')"; \
	fi
	@if [ -f data/test.parquet ]; then \
		echo "  ✓ test.parquet exists"; \
	else \
		echo "  ✗ test.parquet missing (run 'make generate-data')"; \
	fi

## Utilities

tensorboard: ## Start TensorBoard
	@echo "${GREEN}Starting TensorBoard...${RESET}"
	tensorboard --logdir=./runs

build: ## Build package distribution
	@echo "${GREEN}Building package...${RESET}"
	python -m build

clean: ## Clean generated files
	@echo "${GREEN}Cleaning...${RESET}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .pytest_cache/ .coverage
	@echo "${GREEN}Cleaned!${RESET}"

clean-all: clean ## Clean everything including checkpoints and logs
	@echo "${GREEN}Cleaning all generated files...${RESET}"
	rm -rf checkpoints/ runs/ analysis_output/
	@echo "${GREEN}All cleaned!${RESET}"

## CI/CD

ci-check: validate lint-strict test ## Run all CI checks locally
	@echo "${GREEN}All CI checks passed!${RESET}"

pre-commit: validate format lint ## Run before committing
	@echo "${GREEN}Pre-commit checks completed!${RESET}"
