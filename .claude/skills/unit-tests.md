# Unit Tests Skill

Run and analyze unit tests (pytest) for the survival modeling framework.

## Purpose

This skill executes unit tests with pytest, analyzes results, identifies failures,
and helps maintain high test coverage (>80%).

**For end-to-end pipeline testing, use the `integration-tests` skill instead.**

## Usage

This skill focuses on fast, isolated unit tests that verify individual functions
and modules in isolation.

## Commands

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage
```

### Run Specific Tests
```bash
# Run specific test file
pytest tests/test_data.py

# Run specific test function
pytest tests/test_data.py::test_split_X_y

# Run tests by marker
pytest -m unit              # Unit tests only
pytest -m "not slow"        # Skip slow tests
pytest -m integration       # Integration tests only
```

### Run with Parallel Execution
```bash
pytest -n auto
```

### Run with Verbose Output
```bash
pytest -vv
```

## Test Analysis

When tests fail, the skill will:

1. Read the full error traceback
2. Identify the root cause (assertion failure, exception, etc.)
3. Determine if the failure is in:
   - Test code (test needs fixing)
   - Production code (bug in implementation)
   - Test data (fixture or sample data issue)
4. Provide clear explanation of the failure
5. Suggest specific fixes

## Test Categories

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests across modules
- `@pytest.mark.slow` - Time-consuming tests
- `@pytest.mark.requires_data` - Tests requiring sample CSV

## Coverage Requirements

Maintains >80% test coverage:

```bash
# Check current coverage
pytest --cov=src/survival_framework --cov-report=term-missing

# Generate HTML report
pytest --cov=src/survival_framework --cov-report=html
# Open data/test_outputs/coverage/index.html
```

## Test Output Management

All test outputs go to `data/test_outputs/`:
- `coverage/` - HTML coverage reports
- `artifacts/` - Test artifacts (models, predictions)
- `models/` - Test model files

## Code Quality

The skill also verifies code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Test Fixtures

Available fixtures from `conftest.py`:
- `sample_data` - Full sample CSV (1,546 rows)
- `small_sample_data` - First 100 rows for fast tests
- `sample_structured_y` - Small structured survival array
- `temp_artifacts_dir` - Temporary directory for test artifacts
- `test_output_dir` - Persistent test output directory

## Test-Driven Development

Follows TDD principles:

1. **Red**: Write a failing test first
2. **Green**: Write minimal code to make test pass
3. **Refactor**: Clean up code while keeping tests passing

## Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use descriptive names that explain what is being tested

## Arrange-Act-Assert Pattern

```python
def test_example():
    # Arrange: Set up test data
    data = create_test_data()

    # Act: Execute the function
    result = function_under_test(data)

    # Assert: Verify the outcome
    assert result == expected_value
```

## Continuous Improvement

- Add tests for reported bugs before fixing
- Increase coverage for low-coverage modules
- Refactor tests to reduce duplication
- Keep test execution time reasonable (<2 min for full suite)
- Update test documentation when behavior changes
