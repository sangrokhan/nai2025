# Testing Guide

## Overview

This project includes comprehensive testing infrastructure to ensure code quality without requiring dependencies installed.

## Quick Start

### Without Dependencies (Code Validation Only)

```bash
# Run all validation checks
python3 check_all.py

# Or using Makefile
make validate

# Quick syntax check
python3 validate_code.py
make check
```

### With Dependencies (Full Testing)

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test suites
make test-unit          # Unit tests only
make test-integration   # Integration tests only
```

## Test Structure

```
tests/
├── test_models.py       # Model component tests
├── test_data.py         # Data processing tests
├── test_training.py     # Training logic tests
└── test_integration.py  # End-to-end tests
```

## Validation Scripts

### 1. `validate_code.py`

Validates Python syntax and structure:
- ✓ Syntax errors
- ✓ AST parsing
- ✓ Class/function counts

```bash
python3 validate_code.py
```

### 2. `check_all.py`

Comprehensive validation:
- ✓ Python syntax
- ✓ Import structure
- ✓ File structure
- ✓ Code statistics
- ✓ YAML configs

```bash
python3 check_all.py
```

### 3. Test Runner Script

```bash
# Run specific test types
./run_tests.sh unit         # Unit tests
./run_tests.sh integration  # Integration tests
./run_tests.sh models       # Model tests only
./run_tests.sh coverage     # With coverage report
```

## Makefile Commands

```bash
# Development
make install          # Install package
make install-dev      # Install with dev dependencies
make validate         # Run validation (no deps required)
make check           # Quick syntax check

# Testing
make test            # Run all tests
make test-unit       # Unit tests only
make test-integration # Integration tests
make test-cov        # With coverage report

# Code Quality
make format          # Format code with black/isort
make lint            # Lint with flake8
make lint-strict     # Strict linting

# CI/CD
make ci-check        # Run all CI checks locally
make pre-commit      # Run before committing

# Utilities
make clean           # Clean generated files
make clean-all       # Clean everything
```

## GitHub Actions CI

### Automatic PR Checks

When you create a Pull Request, GitHub Actions automatically runs:

1. **Code Validation** (`ci.yml`)
   - Python syntax check
   - Comprehensive validation
   - File structure verification

2. **Test with Dependencies** (`ci.yml`)
   - Install dependencies
   - Test imports
   - Run unit tests
   - Run integration tests

3. **Code Quality** (`ci.yml`)
   - Black formatting check
   - isort import sorting
   - flake8 linting

4. **Build Check** (`ci.yml`)
   - Package build verification

5. **PR Validation** (`pr-validation.yml`)
   - PR title check
   - Test file coverage
   - Documentation check
   - Code metrics
   - Common issues scan

### Workflow Files

```
.github/workflows/
├── ci.yml              # Main CI pipeline
└── pr-validation.yml   # PR-specific checks
```

## Running Tests Locally Before PR

```bash
# Quick validation (no dependencies)
make validate

# Full CI check (requires dependencies)
make ci-check

# Pre-commit checks
make pre-commit
```

## Test Coverage

Generate HTML coverage report:

```bash
make test-cov
# Open htmlcov/index.html in browser
```

## Writing Tests

### Unit Test Example

```python
def test_physical_encoder():
    """Test physical layer encoder"""
    encoder = PhysicalLayerEncoder(channel_dim=64)
    batch = create_dummy_batch()

    h_channel = encoder(batch)

    assert h_channel.shape == (batch_size, 64)
```

### Integration Test Example

```python
def test_full_pipeline():
    """Test complete pipeline"""
    # Preprocess data
    preprocessor = CellularDataPreprocessor()
    processed = preprocessor.fit_transform(df)

    # Create model
    model = HierarchicalModel()

    # Forward pass
    outputs = model(batch)

    # Validate
    assert 'throughput_pred' in outputs
```

## Continuous Integration Status

Check the status of your PR:
- ✅ All checks passing: Ready to merge
- ⚠️  Some warnings: Review and consider fixing
- ❌ Checks failed: Must fix before merge

## Debugging Test Failures

### Syntax Errors

```bash
python3 validate_code.py
# Shows file and line number of syntax errors
```

### Import Errors

```bash
python3 check_all.py
# Shows missing __init__.py or structure issues
```

### Test Failures

```bash
pytest tests/test_models.py -v -s
# -v: verbose
# -s: show print statements
```

### Specific Test

```bash
pytest tests/test_models.py::test_physical_encoder -v
```

## Best Practices

1. **Before Committing**
   ```bash
   make pre-commit
   ```

2. **Before Creating PR**
   ```bash
   make ci-check
   ```

3. **After Changing Models**
   ```bash
   make test-unit
   ```

4. **After Changing Data Pipeline**
   ```bash
   pytest tests/test_data.py tests/test_integration.py -v
   ```

## CI/CD Pipeline Flow

```
PR Created/Updated
    ↓
Code Validation (no deps)
    ├─ Syntax check ✓
    ├─ Import structure ✓
    └─ File structure ✓
    ↓
Test with Dependencies
    ├─ Install deps ✓
    ├─ Import tests ✓
    ├─ Unit tests ✓
    └─ Integration tests ✓
    ↓
Code Quality
    ├─ Black formatting ✓
    ├─ isort imports ✓
    └─ flake8 linting ✓
    ↓
PR Validation
    ├─ PR title ✓
    ├─ Test coverage ✓
    ├─ Documentation ✓
    └─ Code metrics ✓
    ↓
Build Check
    └─ Package build ✓
    ↓
✅ All Checks Passed!
```

## Troubleshooting

### "No module named 'torch'"

You need dependencies for actual tests:
```bash
pip install -r requirements.txt
```

For validation only (no deps):
```bash
python3 check_all.py
```

### Tests Pass Locally but Fail in CI

Check Python version:
```bash
python --version  # Should be 3.8+
```

Check dependencies:
```bash
pip list
```

### Slow Tests

Mark slow tests:
```python
@pytest.mark.slow
def test_large_integration():
    ...
```

Skip them:
```bash
pytest -m "not slow"
```

## Contact

For issues with testing infrastructure, please open a GitHub issue.
