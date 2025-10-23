# Code Quality Guide

This document describes the code quality tools and pre-commit hooks configured for the survival framework.

## Quick Start

```bash
# Install development dependencies (includes pre-commit)
pip install -r requirements-dev.txt

# Setup pre-commit hooks
./scripts/setup-pre-commit.sh

# Or manually:
pre-commit install
```

Once installed, pre-commit hooks will run automatically on `git commit`.

## Manual Checks

### Run All Checks

```bash
# Run all quality checks
./scripts/run-checks.sh
```

### Auto-Format Code

```bash
# Auto-format with black and isort
./scripts/format-code.sh
```

### Individual Tools

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check formatting (without modifying)
black --check src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/

# Static analysis with pylint
pylint src/

# Security checks with bandit
bandit -r src/

# Check docstring coverage
interrogate -vv src/
```

## Pre-commit Hooks

### What Runs on Commit

Pre-commit automatically runs the following checks before each commit:

1. **General File Checks**
   - Remove trailing whitespace
   - Fix end-of-file
   - Check YAML/JSON/TOML syntax
   - Detect large files (>1MB)
   - Detect private keys
   - Check for merge conflicts

2. **Code Formatting**
   - **Black**: Auto-formats Python code (line length: 100)
   - **isort**: Sorts imports automatically

3. **Linting**
   - **Flake8**: Checks code style and complexity
   - **Pylint**: Static code analysis

4. **Type Checking**
   - **Mypy**: Static type checking

5. **Documentation**
   - **Interrogate**: Checks docstring coverage (>80%)

6. **Security**
   - **Bandit**: Scans for common security issues

### Skipping Hooks

To skip pre-commit hooks temporarily (not recommended):

```bash
git commit --no-verify -m "message"
```

### Running Pre-commit Manually

```bash
# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Run on staged files only
pre-commit run
```

### Updating Hooks

```bash
# Update hooks to latest versions
pre-commit autoupdate

# Clean and reinstall hooks
pre-commit clean
pre-commit install
```

## Tool Configurations

### Black (Code Formatting)

**Configuration:** `pyproject.toml` → `[tool.black]`

- Line length: 100 characters
- Python 3.11 target
- Auto-formats on commit

**Settings:**
- Strings use double quotes
- Trailing commas in multi-line structures
- Parentheses in logical operations

### isort (Import Sorting)

**Configuration:** `pyproject.toml` → `[tool.isort]`

- Compatible with Black
- Line length: 100
- Groups: stdlib, third-party, first-party

**Import Order:**
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import pandas as pd

# Local
from survival_framework.data import split_X_y
```

### Flake8 (Linting)

**Configuration:** `.pre-commit-config.yaml`

- Max line length: 100
- Ignored rules: E203, W503, E501
- Max complexity: 10

**Common Checks:**
- Unused imports
- Undefined variables
- Code complexity
- Style violations

### Mypy (Type Checking)

**Configuration:** `pyproject.toml` → `[tool.mypy]`

- Python 3.11
- Ignore missing imports (for external libraries)
- Check untyped definitions

**Usage:**
```python
# Add type hints
def function(param: str) -> int:
    return len(param)

# For complex types
from typing import List, Tuple, Optional

def process(data: List[float]) -> Tuple[float, float]:
    return min(data), max(data)
```

### Pylint (Static Analysis)

**Configuration:** `pyproject.toml` → `[tool.pylint]`

- Max line length: 100
- Disabled rules: C0111, R0903, C0103
- Max arguments: 8

**Checks:**
- Code smells
- Potential bugs
- Unused code
- Code complexity

### Bandit (Security)

**Configuration:** `pyproject.toml` → `[tool.bandit]`

**Checks for:**
- Hard-coded passwords
- SQL injection vulnerabilities
- Use of weak cryptography
- Unsafe file operations
- Command injection risks

### Interrogate (Docstring Coverage)

**Configuration:** `pyproject.toml` → `[tool.interrogate]`

- Required coverage: >80%
- Checks all functions, classes, and modules

**Docstring Requirements:**
```python
def function(param: str) -> int:
    """Calculate length of string.

    Args:
        param: Input string

    Returns:
        Length of the string
    """
    return len(param)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run pre-commit
        run: pre-commit run --all-files

      - name: Run tests with coverage
        run: pytest --cov=src --cov-fail-under=80
```

## Best Practices

### Before Committing

1. Run auto-formatters:
   ```bash
   ./scripts/format-code.sh
   ```

2. Run all checks:
   ```bash
   ./scripts/run-checks.sh
   ```

3. Fix any issues reported

4. Commit your changes

### During Development

- Use type hints for function signatures
- Write docstrings for all public functions
- Keep functions small and focused
- Follow PEP 8 style guide
- Run tests frequently

### Code Review Checklist

- [ ] All pre-commit hooks pass
- [ ] Tests pass with >80% coverage
- [ ] No pylint or flake8 warnings
- [ ] Docstrings present and accurate
- [ ] Type hints added where appropriate
- [ ] No security issues reported by bandit

## Troubleshooting

### Pre-commit Installation Issues

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Update hooks
pre-commit autoupdate
```

### Hook Fails on Commit

1. Read the error message carefully
2. Run the failing hook manually:
   ```bash
   pre-commit run <hook-name> --all-files
   ```
3. Fix the issue
4. Try committing again

### Black and Flake8 Conflicts

Black and Flake8 are configured to work together. If conflicts occur:

1. Black takes precedence for formatting
2. Flake8 ignores Black-compatible rules (E203, W503)
3. Both use 100 character line length

### Slow Pre-commit Runs

First run is slow (installs environments). Subsequent runs are fast.

To speed up:
```bash
# Run in parallel
pre-commit run --all-files --color=always | parallel

# Skip slow checks during development
SKIP=pylint,mypy git commit -m "WIP: quick commit"
```

## Additional Resources

- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 8 Style Guide](https://pep8.org/)
