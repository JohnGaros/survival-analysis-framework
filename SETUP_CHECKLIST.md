# Project Setup Checklist

This checklist guides you through setting up a new data science project with the complete Claude Code infrastructure for **automated, bug-free agentic coding**.

**Time required:** ~30 minutes
**Outcome:** Fully configured project with testing, quality checks, and Claude Code integration

---

## Prerequisites

- [ ] Python 3.11+ installed
- [ ] Git initialized (`git init`)
- [ ] Virtual environment created (`python -m venv .venv`)
- [ ] Virtual environment activated (`source .venv/bin/activate`)

---

## Phase 1: Directory Structure (5 minutes)

### Step 1: Create Directory Tree

```bash
# From project root
mkdir -p src/{project_name}
mkdir -p tests
mkdir -p scripts
mkdir -p data/{sample,raw,processed,test_outputs}
mkdir -p artifacts
mkdir -p models
mkdir -p docs
mkdir -p .claude/agents
```

**Verify:**
```bash
tree -L 2 -d
# Should show:
# .
# ├── .claude
# │   └── agents
# ├── artifacts
# ├── data
# │   ├── processed
# │   ├── raw
# │   ├── sample
# │   └── test_outputs
# ├── docs
# ├── models
# ├── scripts
# ├── src
# │   └── project_name
# └── tests
```

**Status:** ☐ Directories created

---

## Phase 2: Configuration Files (10 minutes)

### Step 2: Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/
.hypothesis/

# Data
data/raw/
data/processed/
*.csv
*.parquet
*.h5
*.hdf5

# Artifacts
artifacts/
models/
*.joblib
*.pkl
*.model

# Test outputs
data/test_outputs/
*.log

# MLflow
mlruns/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Pre-commit
.pre-commit-config.yaml.bak
EOF
```

**Status:** ☐ .gitignore created

---

### Step 3: Create pytest.ini

```bash
cat > pytest.ini << 'EOF'
[pytest]
# Test discovery patterns
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Output options
addopts =
    --verbose
    --strict-markers
    --tb=short
    --disable-warnings
    -ra
    --cov=src/PROJECT_NAME
    --cov-report=term-missing
    --cov-report=html:data/test_outputs/coverage
    --cov-report=xml:data/test_outputs/coverage.xml

# Markers for categorizing tests
markers =
    unit: Unit tests for individual functions
    integration: Integration tests across modules
    slow: Tests that take significant time to run
    requires_data: Tests that require sample data file

# Coverage options
[coverage:run]
source = src/PROJECT_NAME
omit =
    */tests/*
    */__pycache__/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:
EOF

# Replace PROJECT_NAME with your actual project name
sed -i '' 's/PROJECT_NAME/your_project_name/g' pytest.ini
```

**Status:** ☐ pytest.ini created

---

### Step 4: Create pyproject.toml

```bash
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true

[tool.pylint.messages_control]
max-line-length = 100
disable = [
    "C0111",  # missing-docstring
    "R0903",  # too-few-public-methods
    "C0103",  # invalid-name
]

[tool.pylint.format]
max-line-length = 100
max-args = 8

[tool.bandit]
exclude_dirs = ["tests", "scripts"]
skips = ["B101", "B601"]

[tool.interrogate]
fail-under = 80
verbose = 2
ignore-init-module = true
ignore-init-method = true
ignore-magic = true
ignore-module = false
ignore-private = true
EOF
```

**Status:** ☐ pyproject.toml created

---

### Step 5: Create .pre-commit-config.yaml

```bash
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.11
        args: ['--line-length=100']

  - repo: https://github.com/PyCQA/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--profile=black', '--line-length=100']

  - repo: https://github.com/PyCQA/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503,E501', '--max-complexity=10']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: ['--ignore-missing-imports']

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.6
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']
        additional_dependencies: ['bandit[toml]']
EOF
```

**Status:** ☐ .pre-commit-config.yaml created

---

## Phase 3: Scripts (5 minutes)

### Step 6: Create Utility Scripts

```bash
# setup-pre-commit.sh
cat > scripts/setup-pre-commit.sh << 'EOF'
#!/bin/bash
echo "Installing pre-commit hooks..."
pip install pre-commit
pre-commit install
pre-commit autoupdate
pre-commit run --all-files
echo "✓ Pre-commit hooks installed"
EOF

# run-checks.sh
cat > scripts/run-checks.sh << 'EOF'
#!/bin/bash
set -e
echo "Running code quality checks..."
echo "1. Black (formatting)..."
black --check src/ tests/
echo "2. isort (import sorting)..."
isort --check-only src/ tests/
echo "3. Flake8 (linting)..."
flake8 src/ tests/
echo "4. Mypy (type checking)..."
mypy src/
echo "5. Bandit (security)..."
bandit -r src/
echo "✓ All checks passed"
EOF

# format-code.sh
cat > scripts/format-code.sh << 'EOF'
#!/bin/bash
echo "Formatting code..."
black src/ tests/
isort src/ tests/
echo "✓ Code formatted"
EOF

# run-integration-tests.sh
cat > scripts/run-integration-tests.sh << 'EOF'
#!/bin/bash
set -e
echo "Running integration tests..."
rm -rf artifacts/ models/
PYTHONPATH=src timeout 300 python src/main.py > integration_test.log 2>&1
echo "✓ Integration tests passed"
EOF

# Make scripts executable
chmod +x scripts/*.sh
```

**Status:** ☐ Scripts created and made executable

---

## Phase 4: Claude Code Integration (5 minutes)

### Step 7: Create CLAUDE.md

```bash
cat > CLAUDE.md << 'EOF'
# CLAUDE.md

This file provides guidance to Claude Code when working with this project.

## Project Overview

[REPLACE: Brief description of what this project does]

**Recent Updates:**
- [REPLACE: List major changes]

## Running Commands

### Setup and Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Main Pipeline
```bash
python src/main.py
```

### Testing
```bash
# Unit tests (fast)
pytest -m "not integration" -v

# Integration tests (slow)
./scripts/run-integration-tests.sh

# All tests
pytest && ./scripts/run-integration-tests.sh
```

### Code Quality
```bash
# Format code
./scripts/format-code.sh

# Run all checks
./scripts/run-checks.sh
```

## Architecture

### Module Structure

```
src/project_name/
├── data.py        # Data loading and preprocessing
├── models.py      # Model implementations
├── metrics.py     # Evaluation metrics
├── validation.py  # Cross-validation
├── tracking.py    # Experiment tracking
├── utils.py       # Utility functions
└── train.py       # Training pipeline
```

### Data Flow

[REPLACE: Describe how data flows through your system]

### Key Design Patterns

[REPLACE: Document important patterns you use]

## Important Implementation Notes

### Data Requirements

[REPLACE: Document expected data format, columns, dtypes]

### Common Pitfalls

[REPLACE: Document known issues and solutions]

### Testing Requirements

**Critical:** Both unit and integration tests must pass before committing!

```bash
pytest -m "not integration" -v && ./scripts/run-integration-tests.sh
```

## Docstring Requirements

Use Google-style docstrings:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """One-line summary.

    More detailed description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ErrorType: When this error occurs

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        Expected output
    """
```

## Code Quality Standards

- Line length: 100 characters
- Formatting: Black
- Import sorting: isort
- Linting: Flake8
- Type hints: Mypy
- Docstring coverage: >80%
EOF
```

**Status:** ☐ CLAUDE.md created (needs customization)

---

### Step 8: Create Claude Code Agents

```bash
# Test Runner Agent
cat > .claude/agents/test-runner.md << 'EOF'
# Test Runner Agent

## Purpose
Run unit tests after code changes to catch bugs early.

## When to Use
- After modifying any .py file in src/
- Before committing changes
- During development for fast feedback

## What It Tests
- Individual function correctness
- Edge cases and error handling
- Input validation
- Output format correctness

## How to Run
```bash
PYTHONPATH=src pytest -m "not integration" -v
```

## Success Criteria
- All tests pass
- No new failures introduced
- Coverage remains >80%

## Reporting
Report test results with:
- Number of tests passed/failed
- Any new failures
- Coverage percentage
- Specific error messages if failures occur
EOF

# Integration Tester Agent
cat > .claude/agents/integration-tester.md << 'EOF'
# Integration Test Agent

## Purpose
Validate complete end-to-end pipeline execution.

## When to Use
- After changes to data.py, models.py, validation.py, train.py
- Before committing
- Before creating PR
- After major refactoring

## What It Tests
1. Full pipeline execution completes
2. All expected artifacts generated
3. Metrics in valid ranges
4. No LAPACK/Index/Type errors
5. Preprocessing handles edge cases
6. Models compatible with pipeline

## How to Run
```bash
./scripts/run-integration-tests.sh
```

## Success Criteria
- Pipeline completes without errors
- All CSV files and model files created
- Metrics within expected ranges (0.5 < C-index < 1.0, etc.)
- No critical errors in logs

## Reporting
Report with:
- Pipeline completion status (PASS/FAIL)
- Duration in seconds
- List of artifacts created
- Metric ranges
- Any errors found (LAPACK, IndexError, TypeError)
- Warnings (ConvergenceWarning, overflow, etc.)
EOF
```

**Status:** ☐ Claude agents created

---

## Phase 5: Testing Infrastructure (5 minutes)

### Step 9: Create Test Configuration

```bash
# tests/conftest.py
cat > tests/conftest.py << 'EOF'
"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def sample_data_path(project_root):
    """Path to sample data CSV."""
    return project_root / "data" / "sample" / "sample_data.csv"

@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test artifacts."""
    return tmp_path

@pytest.fixture
def sample_dataframe():
    """Create small sample DataFrame for testing."""
    return pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': ['A', 'B', 'A', 'B', 'A'],
        'target': [0, 1, 0, 1, 0]
    })
EOF

# tests/__init__.py
touch tests/__init__.py

# tests/test_integration.py
cat > tests/test_integration.py << 'EOF'
"""Integration tests for end-to-end pipeline."""

import pytest

pytestmark = pytest.mark.integration

class TestEndToEndPipeline:
    """Test complete pipeline execution."""

    def test_pipeline_completes_successfully(self, sample_data_path):
        """Test that full pipeline completes without errors."""
        # TODO: Import and call main pipeline function
        # from project_name.train import train_pipeline
        # train_pipeline(str(sample_data_path))
        pass
EOF
```

**Status:** ☐ Test infrastructure created

---

## Phase 6: Dependencies (3 minutes)

### Step 10: Create Requirements Files

```bash
# requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Add your project-specific dependencies here
# e.g., torch, tensorflow, transformers, etc.
EOF

# requirements-dev.txt
cat > requirements-dev.txt << 'EOF'
# Include production requirements
-r requirements.txt

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-xdist>=3.3.0

# Code quality
black>=23.7.0
flake8>=6.1.0
flake8-docstrings>=1.7.0
flake8-bugbear>=23.9.0
mypy>=1.5.0
pylint>=2.17.5
isort>=5.12.0
bandit>=1.7.0
interrogate>=1.5.0

# Pre-commit hooks
pre-commit>=3.5.0

# Type stubs
pandas-stubs>=2.0.3
types-setuptools>=68.1.0.0
EOF
```

**Status:** ☐ Requirements files created

---

## Phase 7: Documentation (2 minutes)

### Step 11: Create Documentation Files

```bash
# README.md
cat > README.md << 'EOF'
# [Project Name]

[Brief description of what this project does]

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
./scripts/setup-pre-commit.sh
```

## Usage

```bash
# Run main pipeline
python src/main.py

# Run tests
pytest -m "not integration" -v
./scripts/run-integration-tests.sh
```

## Documentation

- **CLAUDE.md** - Context for Claude Code
- **TESTING.md** - Testing guide
- **CODE_QUALITY.md** - Quality standards
- **PROJECT_TEMPLATE.md** - Template structure
- **SETUP_CHECKLIST.md** - This file

## License

[Your license]
EOF

# Create src package
cat > src/project_name/__init__.py << 'EOF'
"""Project Name - [Brief description]"""

__version__ = "0.1.0"
EOF

# Create basic main.py
cat > src/main.py << 'EOF'
"""Main entry point for the application."""

def main():
    """Main function."""
    print("Hello from project!")

if __name__ == "__main__":
    main()
EOF
```

**Status:** ☐ Documentation created

---

## Phase 8: Installation and Verification (2 minutes)

### Step 12: Install Dependencies

```bash
# Activate virtual environment if not already activated
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Setup pre-commit hooks
./scripts/setup-pre-commit.sh
```

**Status:** ☐ Dependencies installed

---

### Step 13: Verify Setup

```bash
# 1. Check directory structure
tree -L 2 -I '__pycache__|*.pyc'

# 2. Run unit tests (should have at least placeholder tests)
pytest -m "not integration" -v

# 3. Run code quality checks
./scripts/run-checks.sh

# 4. Test pre-commit hooks
pre-commit run --all-files

# 5. Verify scripts are executable
ls -l scripts/*.sh
```

**Expected outcomes:**
- ✓ Directory structure matches template
- ✓ Pytest discovers test files
- ✓ Code quality checks pass (or show no files)
- ✓ Pre-commit hooks installed
- ✓ Scripts have execute permission

**Status:** ☐ Setup verified

---

## Phase 9: Customize for Your Project (Variable time)

### Step 14: Project-Specific Configuration

Now customize the template for your specific project:

- [ ] **CLAUDE.md**: Update project description, architecture, data requirements
- [ ] **pytest.ini**: Replace `PROJECT_NAME` with your actual project name
- [ ] **README.md**: Add project-specific information
- [ ] **requirements.txt**: Add your specific dependencies
- [ ] **src/project_name/**: Rename to your actual project name
- [ ] **tests/**: Add your specific test cases
- [ ] **.gitignore**: Add any project-specific ignore patterns

**Status:** ☐ Project customized

---

### Step 15: Initial Commit

```bash
git add .
git commit -m "Initial project setup with Claude Code infrastructure

- Added directory structure
- Configured testing (pytest, unit + integration)
- Configured code quality (black, isort, flake8, mypy)
- Added pre-commit hooks
- Created Claude Code agents
- Added comprehensive documentation
"
```

**Status:** ☐ Initial commit created

---

## Final Checklist

Before starting development, verify:

- [ ] Directory structure complete
- [ ] All configuration files created
- [ ] Scripts created and executable
- [ ] CLAUDE.md customized for project
- [ ] Claude agents configured
- [ ] Test infrastructure in place
- [ ] Dependencies installed
- [ ] Pre-commit hooks installed
- [ ] Documentation complete
- [ ] Initial commit made

---

## Quick Reference

### Daily Workflow

```bash
# 1. Make changes to code

# 2. Run unit tests (fast)
pytest -m "not integration" -v

# 3. Before committing, run integration tests
./scripts/run-integration-tests.sh

# 4. Both passed? Commit!
git add .
git commit -m "your changes"
```

### Working with Claude Code

```
"Please implement feature X.

Requirements:
1. Read CLAUDE.md for context
2. Add tests to tests/
3. Run unit and integration tests
4. Report results"
```

### Maintenance

```bash
# Update dependencies
pip install --upgrade -r requirements-dev.txt

# Update pre-commit hooks
pre-commit autoupdate

# Run all quality checks
./scripts/run-checks.sh
```

---

## Troubleshooting

### Issue: Pre-commit hooks fail

```bash
# Fix formatting issues automatically
./scripts/format-code.sh

# Run hooks again
pre-commit run --all-files
```

### Issue: Tests not discovered

```bash
# Check PYTHONPATH
export PYTHONPATH=src:$PYTHONPATH

# Run with explicit path
pytest tests/ -v
```

### Issue: Import errors

```bash
# Install in editable mode
pip install -e .

# Or always use PYTHONPATH
PYTHONPATH=src pytest tests/ -v
```

---

## Success!

You now have a fully configured data science project with:

✅ **Complete directory structure**
✅ **Testing infrastructure** (unit + integration)
✅ **Code quality enforcement** (pre-commit hooks)
✅ **Claude Code integration** (CLAUDE.md + agents)
✅ **Comprehensive documentation**
✅ **Automated workflows**

**Next steps:**
1. Customize CLAUDE.md for your project
2. Add your first feature
3. Write tests
4. Let Claude Code help you build! 🚀

**Time to first productive coding:** ~30 minutes from running this checklist!
