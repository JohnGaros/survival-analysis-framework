# Testing Guide

This document provides comprehensive guidance for testing the survival modeling framework.

## Test Strategy Overview

The framework uses **two complementary test approaches**:

| Test Type | Purpose | Speed | When to Run |
|-----------|---------|-------|-------------|
| **Unit Tests** | Test individual functions | Fast (~10s) | Every commit |
| **Integration Tests** | Test end-to-end workflows | Slow (~60-120s) | Before merge/deploy |

**Critical:** Both test types are required. Unit tests catch logic errors, integration tests catch workflow breakage.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run unit tests only (fast)
pytest -m "not integration"

# Run integration tests (slow)
./scripts/run-integration-tests.sh

# Run ALL tests (unit + integration)
pytest && ./scripts/run-integration-tests.sh

# Run with coverage
pytest --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage
```

## Test Structure

### Test Organization

```
tests/
├── conftest.py          # Shared fixtures and pytest configuration
├── test_data.py         # Tests for data loading and preprocessing
├── test_metrics.py      # Tests for survival metrics
├── test_utils.py        # Tests for utility functions
└── test_*.py           # Additional test modules
```

### Test Categories

Tests are marked with categories for selective execution:

- **unit**: Fast unit tests for individual functions
- **integration**: Tests across multiple modules
- **slow**: Tests that take significant time (>1 second)
- **requires_data**: Tests that need the sample CSV file

```bash
# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Run integration tests
pytest -m integration
```

## Writing Tests

### Test Naming Convention

```python
def test_<function_name>_<scenario>_<expected_outcome>():
    """Brief description of what is being tested."""
    pass
```

Examples:
```python
def test_split_X_y_removes_missing_values_when_dropna_true():
    """Test that dropna=True removes rows with NaN values."""
    pass

def test_compute_cindex_returns_value_between_zero_and_one():
    """Test that C-index is in valid range."""
    pass
```

### Test Structure (Arrange-Act-Assert)

```python
def test_example():
    # Arrange: Set up test data and dependencies
    df = pd.DataFrame({'col': [1, 2, 3]})

    # Act: Execute the function under test
    result = process_data(df)

    # Assert: Verify the outcome
    assert result.shape == (3, 1)
    assert result['col'].sum() == 6
```

### Using Fixtures

Fixtures are defined in `tests/conftest.py`:

```python
def test_with_sample_data(sample_data):
    """Use full sample CSV data."""
    assert len(sample_data) > 0
    assert 'survival_months' in sample_data.columns

def test_with_small_sample(small_sample_data):
    """Use first 100 rows for faster tests."""
    assert len(small_sample_data) == 100

def test_with_temp_dir(temp_artifacts_dir):
    """Use temporary directory that is automatically cleaned up."""
    output_file = temp_artifacts_dir / "test.csv"
    # ... use output_file ...
```

### Available Fixtures

- `project_root` - Path to project root directory
- `sample_data_path` - Path to sample CSV file
- `test_output_dir` - Path to test outputs directory
- `sample_data` - Loaded full sample DataFrame
- `small_sample_data` - First 100 rows of sample data
- `sample_structured_y` - Small structured survival array
- `temp_artifacts_dir` - Temporary directory for test artifacts
- `temp_models_dir` - Temporary directory for test models

## Test-Driven Development (TDD)

### TDD Workflow

1. **Red**: Write a failing test
```python
def test_new_feature():
    """Test for feature that doesn't exist yet."""
    result = new_feature(input_data)
    assert result == expected_output  # This will fail
```

2. **Green**: Write minimal code to pass the test
```python
def new_feature(data):
    """Minimal implementation."""
    return expected_output
```

3. **Refactor**: Improve code while keeping tests passing
```python
def new_feature(data):
    """Improved implementation with better structure."""
    # Refactored code that still passes the test
    return processed_result
```

### TDD Best Practices

- Write tests before implementation
- Keep tests simple and focused
- Test one thing per test function
- Use descriptive test names
- Make tests independent of each other
- Keep test execution fast

## Coverage Requirements

### Target Coverage

- Overall: >80%
- Critical modules (data.py, models.py): >90%
- Utility modules: >75%

### Checking Coverage

```bash
# Terminal report with missing lines
pytest --cov=src/survival_framework --cov-report=term-missing

# HTML report (more detailed)
pytest --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage
open data/test_outputs/coverage/index.html

# XML report (for CI/CD)
pytest --cov=src/survival_framework --cov-report=xml:data/test_outputs/coverage.xml
```

### Improving Coverage

1. Identify uncovered lines:
```bash
pytest --cov=src/survival_framework --cov-report=term-missing
```

2. Add tests for uncovered code:
```python
def test_edge_case_that_was_missed():
    """Test previously uncovered code path."""
    pass
```

3. Verify improvement:
```bash
pytest --cov=src/survival_framework
```

## Running Tests

### Basic Execution

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Very verbose (show individual assertions)
pytest -vv

# Show print statements
pytest -s
```

### Selective Execution

```bash
# Run specific file
pytest tests/test_data.py

# Run specific test
pytest tests/test_data.py::TestSplitXY::test_basic_split

# Run tests matching pattern
pytest -k "cindex"

# Run tests with specific marker
pytest -m unit

# Combine markers
pytest -m "unit and not slow"
```

### Parallel Execution

```bash
# Auto-detect CPU count
pytest -n auto

# Specify number of workers
pytest -n 4
```

### Debugging Failed Tests

```bash
# Stop at first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Show local variables on failure
pytest -l

# Show full diff for assertions
pytest --tb=long
```

## Continuous Integration

### Pre-commit Checks

Before committing, run:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
flake8 src/ tests/

# Run tests with coverage
pytest --cov=src/survival_framework --cov-report=term-missing

# Type checking
mypy src/
```

### CI Pipeline

Typical CI pipeline should:

1. Install dependencies
2. Run linters (flake8, black, isort)
3. Run type checking (mypy)
4. Run test suite with coverage
5. Fail if coverage < 80%
6. Upload coverage reports

## Common Testing Patterns

### Testing Exceptions

```python
def test_function_raises_value_error_for_invalid_input():
    """Test that ValueError is raised for invalid input."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_fail(invalid_input)
```

### Testing with Mock Data

```python
def test_with_mock_data():
    """Test with artificially created data."""
    y = np.array(
        [(True, 12.0), (False, 24.0)],
        dtype=[("event", bool), ("time", float)]
    )
    result = compute_metric(y)
    assert result > 0
```

### Parametrized Tests

```python
@pytest.mark.parametrize("n_splits,expected", [
    (3, 3),
    (5, 5),
    (10, 10),
])
def test_splitter_creates_correct_number_of_folds(n_splits, expected):
    """Test that splitter creates expected number of folds."""
    cfg = CVConfig(n_splits=n_splits)
    splits = event_balanced_splitter(y_data, cfg)
    assert len(splits) == expected
```

### Testing File I/O

```python
def test_save_metrics_creates_file(temp_artifacts_dir):
    """Test that metrics are saved to file."""
    df = pd.DataFrame({'metric': [0.75]})
    path = save_model_metrics(df, str(temp_artifacts_dir))

    assert os.path.exists(path)
    loaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(df, loaded)
```

## Troubleshooting

### Tests Pass Locally But Fail in CI

- Check Python version consistency
- Verify all dependencies are in requirements-dev.txt
- Check for environment-specific paths
- Ensure tests don't depend on execution order

### Slow Test Suite

- Use pytest-xdist for parallel execution: `pytest -n auto`
- Mark slow tests: `@pytest.mark.slow`
- Use smaller fixtures: `small_sample_data` instead of `sample_data`
- Mock expensive operations
- Profile tests: `pytest --durations=10`

### Flaky Tests

- Check for test interdependencies
- Verify random seeds are set
- Look for timing-dependent assertions
- Ensure fixtures are properly isolated

### Import Errors

- Verify PYTHONPATH includes project root
- Check that `__init__.py` files exist
- Ensure modules are installed: `pip install -e .`

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Test-Driven Development Guide](https://testdriven.io/)
- Framework-specific guidance in `.claude/agents/test-runner.md`

## Integration Testing

### Why Integration Tests Are Critical

**Unit tests passed but end-to-end failed** - This is a common problem! Here's why:

| What Unit Tests Miss | How Integration Tests Catch It |
|---------------------|--------------------------------|
| Pipeline compatibility | Tests `VarianceThreshold` → model interaction |
| Data edge cases | Uses real 2000-row dataset with zero times, missing values |
| Cross-validation issues | Tests fold-specific time constraints |
| Model prediction compatibility | Tests `predict_survival_function()` on all models |
| Artifact generation | Verifies all CSV files and model files created |

### Integration Test Suite

**Location:** `tests/test_integration.py`

**Coverage:**
1. ✅ Full pipeline execution (`train_all_models()`)
2. ✅ All 5 models complete training
3. ✅ All 5 CV folds complete per model
4. ✅ All artifacts generated (CSV, model files, predictions)
5. ✅ Metrics in valid ranges (C-index, IBS)
6. ✅ Preprocessing handles missing values, zero variance, rare categories
7. ✅ Model compatibility with pipeline output

### Running Integration Tests

**Option 1: Shell Script (Recommended)**
```bash
./scripts/run-integration-tests.sh
```

This script:
- Cleans old artifacts
- Runs full training pipeline
- Verifies all outputs
- Validates metrics
- Checks for critical errors
- Generates detailed report

**Option 2: Pytest**
```bash
# Run integration tests via pytest
pytest -m integration -v

# Run specific integration test
pytest tests/test_integration.py::TestEndToEndPipeline -v
```

### Integration Test Output

```bash
=========================================
Integration Test Suite
=========================================

Step 1: Clean environment
-------------------------
✓ Cleaned artifacts and test outputs

Step 2: Run end-to-end pipeline
--------------------------------
✓ Pipeline completed successfully (89s)

Step 3: Verify artifacts
------------------------
Testing: model_metrics.csv exists ... ✓ PASS
Testing: model_summary.csv exists ... ✓ PASS
Testing: cox_ph artifacts exist ... ✓ PASS
Testing: coxnet artifacts exist ... ✓ PASS
Testing: Models saved (5+ expected) ... ✓ PASS (found 5)

Step 4: Validate metrics
------------------------
All 5 models present: cox_ph, coxnet, gbsa, rsf, weibull_aft
C-index range: [0.863, 0.895]
IBS range: [0.074, 0.103]
✓ Metrics validation passed

Step 5: Check for critical errors
----------------------------------
✓ No LAPACK errors
✓ No IndexErrors
✓ No TypeErrors

=========================================
  ✓ ALL INTEGRATION TESTS PASSED
=========================================
```

### When to Run Integration Tests

| Scenario | Unit Tests | Integration Tests |
|----------|-----------|------------------|
| During development | ✅ Every save | ❌ Not needed |
| Before committing | ✅ Required | ✅ **Required** |
| After changing data.py | ✅ Required | ✅ **Critical** |
| After changing models.py | ✅ Required | ✅ **Critical** |
| After changing validation.py | ✅ Required | ✅ **Critical** |
| Before creating PR | ✅ Required | ✅ **Required** |
| In CI/CD pipeline | ✅ Always | ✅ Always |

### Integration Test Checklist

Before committing changes:

```bash
# 1. Run unit tests
pytest -m "not integration" -v

# 2. If unit tests pass, run integration tests
./scripts/run-integration-tests.sh

# 3. Both passed? Safe to commit
git add .
git commit -m "feat: your changes"
```

**If integration tests fail but unit tests passed:**
- Check `integration_test.log` for errors
- Look for LAPACK, IndexError, or TypeError messages
- Verify preprocessing pipeline compatibility
- Check that all models can handle numpy array inputs
- Ensure time grid constraints are applied

### Common Integration Test Failures

#### 1. LAPACK Error
```
ValueError: LAPACK reported an illegal value
```
**Cause:** Missing value handling, zero variance, or no regularization  
**Fix:** Check preprocessing pipeline has imputation and variance filtering

#### 2. Time Range Error
```
ValueError: all times must be within follow-up time
```
**Cause:** Time points exceed test fold range  
**Fix:** Verify time grid constraints in `validation.py`

#### 3. IndexError
```
IndexError: only integers, slices (...) are valid indices
```
**Cause:** Model expects DataFrame but receives numpy array  
**Fix:** Add array-to-DataFrame conversion in model's `fit()` method

#### 4. TypeError
```
TypeError: DataFrame expected, got ndarray
```
**Cause:** lifelines models require DataFrames after VarianceThreshold  
**Fix:** Convert numpy array to DataFrame with generic column names

### Creating New Integration Tests

```python
import pytest

@pytest.mark.integration
class TestMyNewFeature:
    """Integration tests for new feature."""

    def test_feature_works_end_to_end(self, sample_data_path):
        """Test new feature in complete pipeline."""
        from survival_framework.train import train_all_models

        # This should not raise any exceptions
        train_all_models(str(sample_data_path))

        # Verify specific outputs
        assert Path("artifacts/my_feature_output.csv").exists()
```

**Key Points:**
- Use `@pytest.mark.integration` decorator
- Test with real sample data (2000 rows)
- Verify end-to-end workflow, not isolated functions
- Check artifact generation
- Validate output files exist and contain expected data

## Continuous Integration Setup

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run unit tests
        run: pytest -m "not integration" -v

      - name: Run integration tests
        run: ./scripts/run-integration-tests.sh

      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: |
            integration_test.log
            data/test_outputs/coverage/
```

## Best Practices Summary

### ✅ Do

1. **Run both unit and integration tests before committing**
2. **Write integration tests for new models**
3. **Test with real sample data (2000 rows)**
4. **Verify all artifacts are generated**
5. **Check for LAPACK, Index, and Type errors**
6. **Use integration tests to catch workflow bugs**

### ❌ Don't

1. **Don't skip integration tests** - They catch pipeline issues unit tests miss
2. **Don't use mock data for integration tests** - Use real sample data
3. **Don't commit if integration tests fail** - Fix the issue first
4. **Don't rely solely on unit tests** - They don't test end-to-end workflows
5. **Don't ignore warnings** - They may indicate serious issues

## Troubleshooting

### Integration Tests Fail But Unit Tests Pass

This is expected! It means:
- ✅ Individual functions work correctly (unit tests)
- ❌ Functions don't work together in pipeline (integration tests)

**Common causes:**
1. Preprocessing output incompatible with model input
2. Pipeline steps modify data structure (DataFrame → numpy array)
3. Time grid constraints not applied in CV folds
4. Model can't handle edge cases (zero times, rare categories)

**Solution:**
1. Check `integration_test.log` for specific error
2. Run failing test in isolation: `pytest tests/test_integration.py::<test_name> -v`
3. Debug the pipeline step causing the issue
4. Fix and re-run both unit and integration tests

### Integration Tests Are Slow

Yes, they are! (60-120 seconds)

**Why:**
- Trains 5 models × 5 folds = 25 model fits
- Uses full 2000-row dataset
- Tests complete end-to-end workflow

**Solutions:**
- Run unit tests during development (fast feedback)
- Run integration tests before commit/merge (thorough validation)
- Use `pytest -m "not slow"` to skip during development
- Let CI/CD run integration tests automatically

## Summary

**Critical Insight:** Unit tests verify functions work in isolation. Integration tests verify functions work together in the complete pipeline.

**Always run BOTH before committing!**

```bash
# Complete test workflow
pytest -m "not integration" -v  # Fast: ~10s
./scripts/run-integration-tests.sh  # Slow: ~90s

# Both passed? Safe to push!
git push
```
