#!/bin/bash
# Test Runner Script
# Convenient wrapper for running different types of tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Cellular Network Optimization - Test Suite ===${NC}\n"

# Parse arguments
TEST_TYPE=${1:-all}

case $TEST_TYPE in
    "unit")
        echo -e "${YELLOW}Running unit tests only...${NC}"
        pytest tests/test_models.py tests/test_data.py tests/test_training.py -v
        ;;
    "integration")
        echo -e "${YELLOW}Running integration tests...${NC}"
        pytest tests/test_integration.py -v
        ;;
    "models")
        echo -e "${YELLOW}Running model tests...${NC}"
        pytest tests/test_models.py -v
        ;;
    "data")
        echo -e "${YELLOW}Running data tests...${NC}"
        pytest tests/test_data.py -v
        ;;
    "training")
        echo -e "${YELLOW}Running training tests...${NC}"
        pytest tests/test_training.py -v
        ;;
    "coverage")
        echo -e "${YELLOW}Running tests with coverage...${NC}"
        pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
        echo -e "${GREEN}Coverage report saved to htmlcov/index.html${NC}"
        ;;
    "quick")
        echo -e "${YELLOW}Running quick tests (excluding slow tests)...${NC}"
        pytest tests/ -v -m "not slow"
        ;;
    "all")
        echo -e "${YELLOW}Running all tests...${NC}"
        pytest tests/ -v
        ;;
    "help")
        echo "Usage: ./run_tests.sh [TEST_TYPE]"
        echo ""
        echo "TEST_TYPE options:"
        echo "  all         - Run all tests (default)"
        echo "  unit        - Run only unit tests"
        echo "  integration - Run only integration tests"
        echo "  models      - Run model tests"
        echo "  data        - Run data tests"
        echo "  training    - Run training tests"
        echo "  coverage    - Run tests with coverage report"
        echo "  quick       - Run quick tests (exclude slow tests)"
        echo "  help        - Show this help message"
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✓ Tests completed successfully!${NC}"
