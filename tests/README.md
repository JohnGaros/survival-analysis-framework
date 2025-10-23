# tests/

## Purpose

Comprehensive test suite for survival modeling framework.

## Contents

- `conftest.py` - Shared pytest fixtures and configuration
- `test_data.py` - Unit tests for data loading and preprocessing
- `test_metrics.py` - Unit tests for survival metrics
- `test_utils.py` - Unit tests for utility functions
- `test_integration.py` - End-to-end pipeline integration tests

## Test Strategy

**Two-tier approach:**
1. **Unit Tests** - Fast (~10s), test individual functions in isolation
2. **Integration Tests** - Slow (~90s), test complete pipeline execution

## Running Tests

```bash
# Unit tests only (fast)
pytest -m "not integration" -v

# Integration tests (slow)
./scripts/run-integration-tests.sh

# All tests with coverage
pytest --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage
```

## Critical

**Both test tiers must pass before committing!** Unit tests verify function correctness, integration tests verify pipeline compatibility.

See `TESTING.md` for complete testing guide.
