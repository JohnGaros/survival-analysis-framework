# Data Science Project Template with Claude Code

This document describes the complete project structure and Claude Code infrastructure that ensures **automated, bug-free agentic coding**. Use this as a template for new data science projects.

---

## Philosophy

**Goal:** Enable Claude Code to work autonomously with:
1. ✅ **Sufficient context** - Knows architecture, conventions, and patterns
2. ✅ **Automated testing** - Catches bugs before they reach production
3. ✅ **Quality enforcement** - Pre-commit hooks prevent bad code
4. ✅ **Self-contained agents** - Specialized agents for specific tasks

**Key Principle:** "Read one file → Understand entire project setup"

---

## Project Structure

### Complete Directory Tree

```
project_name/
├── .claude/                      # Claude Code configuration
│   └── agents/                   # Specialized agents
│       ├── test-runner.md        # Runs unit tests
│       └── integration-tester.md # Runs end-to-end tests
│
├── src/                          # Source code
│   └── project_name/             # Main package
│       ├── __init__.py
│       ├── data.py               # Data loading and preprocessing
│       ├── models.py             # Model implementations
│       ├── metrics.py            # Evaluation metrics
│       ├── validation.py         # Cross-validation logic
│       ├── tracking.py           # Experiment tracking (MLflow, etc.)
│       ├── utils.py              # Utility functions
│       └── train.py              # Training pipeline
│
├── tests/                        # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_data.py             # Unit tests for data.py
│   ├── test_models.py           # Unit tests for models.py
│   ├── test_metrics.py          # Unit tests for metrics.py
│   ├── test_utils.py            # Unit tests for utils.py
│   └── test_integration.py      # Integration tests
│
├── scripts/                      # Utility scripts
│   ├── setup-pre-commit.sh      # Install pre-commit hooks
│   ├── run-checks.sh            # Run all quality checks
│   ├── format-code.sh           # Auto-format code
│   ├── run-integration-tests.sh # Run end-to-end tests
│   └── README.md                # Script documentation
│
├── data/                         # Data directory
│   ├── sample/                  # Sample data for testing
│   ├── raw/                     # Raw input data (gitignored)
│   ├── processed/               # Processed data (gitignored)
│   └── test_outputs/            # Test artifacts (gitignored)
│
├── artifacts/                    # Training artifacts (gitignored)
│   ├── model_metrics.csv        # Model performance metrics
│   ├── model_summary.csv        # Aggregated results
│   └── {model_name}/            # Per-model artifacts
│
├── models/                       # Trained models (gitignored)
│   └── *.joblib                 # Serialized models
│
├── docs/                         # Additional documentation
│   └── architecture.md          # Architecture diagrams
│
├── .gitignore                    # Git ignore rules
├── .pre-commit-config.yaml       # Pre-commit hook config
├── pytest.ini                    # Pytest configuration
├── pyproject.toml                # Tool configurations (black, mypy, etc.)
│
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
│
├── CLAUDE.md                     # ⭐ Main Claude Code context file
├── README.md                     # Project overview
├── TESTING.md                    # Testing guide
├── CODE_QUALITY.md               # Code quality guide
├── SETUP_CHECKLIST.md            # Quick setup guide
└── PROJECT_TEMPLATE.md           # This file
```

---

## Essential Files for Claude Code

### 1. **CLAUDE.md** (Most Important)

**Purpose:** Single source of truth for Claude Code
**Location:** Project root
**Content:**

```markdown
# CLAUDE.md

## Project Overview
[Brief description of what the project does]

**Recent Updates:**
- [Key changes and fixes]

## Running Commands

### Setup and Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Training/Main Pipeline
```bash
python src/main.py
```

### Testing
```bash
# Unit tests
pytest -m "not integration" -v

# Integration tests
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
[Describe main modules and their responsibilities]

### Data Flow
[Describe how data flows through the system]

### Key Design Patterns
[Document important patterns and conventions]

## Important Implementation Notes

### Data Requirements
[Document expected data format, column names, dtypes]

### Common Pitfalls
[Document known issues and how to avoid them]

### Testing Requirements
[Document that both unit and integration tests must pass]

## Docstring Requirements
[Document expected docstring format - Google, NumPy, etc.]
```

**Why it's critical:** Claude reads this first to understand the project

---

### 2. **.claude/agents/** Directory

**Purpose:** Specialized agents for specific tasks
**Structure:**

```
.claude/
└── agents/
    ├── test-runner.md           # Runs unit tests after changes
    ├── integration-tester.md    # Runs end-to-end tests
    ├── code-reviewer.md         # Reviews code quality
    ├── documentation-writer.md  # Updates documentation
    └── deployment-checker.md    # Validates deployment readiness
```

**Example Agent:** `.claude/agents/integration-tester.md`

```markdown
# Integration Test Agent

## Purpose
Validates complete end-to-end pipeline execution.

## When to Use
- After changes to data.py, models.py, validation.py, train.py
- Before committing
- Before creating PR

## What It Tests
1. Full pipeline execution (train_all_models())
2. All models complete training
3. All artifacts generated
4. Metrics in valid ranges
5. No LAPACK/Index/Type errors

## How to Use
```bash
./scripts/run-integration-tests.sh
```

## Success Criteria
- Pipeline completes without errors
- All expected artifacts created
- Metrics in valid ranges
- No critical errors in logs
```

**Why it's critical:** Automates testing after code changes

---

### 3. **TESTING.md**

**Purpose:** Complete testing guide
**Sections:**

1. **Test Strategy Overview** - Unit vs integration tests
2. **Quick Start** - Commands to run tests
3. **Test Structure** - How tests are organized
4. **Writing Tests** - Conventions and examples
5. **Integration Testing** - Why it's critical
6. **CI/CD Setup** - Automation examples

**Why it's critical:** Ensures consistent testing practices

---

### 4. **CODE_QUALITY.md**

**Purpose:** Code quality standards and tools
**Sections:**

1. **Quick Start** - Setup commands
2. **Pre-commit Hooks** - What runs automatically
3. **Tool Configurations** - Black, isort, flake8, mypy
4. **Manual Checks** - How to run tools manually
5. **Best Practices** - Coding standards

**Why it's critical:** Enforces code quality automatically

---

### 5. **SETUP_CHECKLIST.md** (New - To Be Created)

**Purpose:** Quick setup guide for new projects
**Content:** Step-by-step instructions to replicate this structure

---

## Configuration Files

### 1. **.pre-commit-config.yaml**

**Purpose:** Automated code quality checks
**Contains:**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/pycqa/isort
    hooks:
      - id: isort
        args: ['--profile=black']

  - repo: https://github.com/pycqa/flake8
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
```

---

### 2. **pytest.ini**

**Purpose:** Pytest configuration
**Contains:**

```ini
[pytest]
python_files = test_*.py
python_classes = Test*
python_functions = test_*

testpaths = tests

addopts =
    --verbose
    --strict-markers
    --tb=short
    --cov=src/project_name
    --cov-report=term-missing
    --cov-report=html:data/test_outputs/coverage

markers =
    unit: Unit tests for individual functions
    integration: Integration tests across modules
    slow: Tests that take significant time to run
```

---

### 3. **pyproject.toml**

**Purpose:** Centralized tool configuration
**Contains:**

```toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

[tool.pylint.messages_control]
max-line-length = 100

[tool.interrogate]
fail-under = 80
verbose = 2
```

---

### 4. **.gitignore**

**Purpose:** Prevent committing artifacts
**Key sections:**

```gitignore
# Data
data/raw/
data/processed/
*.csv
*.parquet

# Artifacts
artifacts/
models/
*.joblib
*.pkl

# Testing
.pytest_cache/
.coverage
htmlcov/
data/test_outputs/

# Python
__pycache__/
*.py[cod]
.venv/
venv/

# MLflow
mlruns/

# IDE
.vscode/
.idea/
```

---

## Testing Infrastructure

### Unit Tests

**Location:** `tests/test_*.py`
**Purpose:** Fast feedback on individual functions
**Example:**

```python
# tests/test_data.py
def test_load_data_returns_dataframe():
    """Test that load_data returns a pandas DataFrame."""
    df = load_data('data/sample/input.csv')
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
```

---

### Integration Tests

**Location:** `tests/test_integration.py`
**Purpose:** Validate end-to-end workflows
**Example:**

```python
# tests/test_integration.py
@pytest.mark.integration
def test_full_pipeline_completes(sample_data_path):
    """Test that complete training pipeline works."""
    train_all_models(str(sample_data_path))

    # Verify artifacts
    assert Path('artifacts/model_metrics.csv').exists()
    assert Path('artifacts/model_summary.csv').exists()
```

---

### Integration Test Script

**Location:** `scripts/run-integration-tests.sh`
**Purpose:** Comprehensive end-to-end validation
**Features:**

- Cleans old artifacts
- Runs full training pipeline
- Verifies all outputs
- Validates metrics
- Checks for errors
- Generates report

---

## Quality Enforcement

### Pre-commit Hooks

**Setup:**
```bash
pip install pre-commit
pre-commit install
```

**What runs on commit:**
1. Trailing whitespace removal
2. Black code formatting
3. isort import sorting
4. Flake8 linting
5. Mypy type checking
6. Bandit security checks

**To skip (not recommended):**
```bash
git commit --no-verify
```

---

### Manual Quality Checks

```bash
# Format code
./scripts/format-code.sh

# Run all checks
./scripts/run-checks.sh
```

---

## Workflow for Claude Code

### When Claude Makes Changes

1. **Read CLAUDE.md** - Understand project context
2. **Make changes** - Implement requested feature
3. **Run unit tests** - Fast validation
   ```bash
   pytest -m "not integration" -v
   ```
4. **Run integration tests** - Thorough validation
   ```bash
   ./scripts/run-integration-tests.sh
   ```
5. **Report results** - Tell user if tests pass/fail

---

### Example Claude Prompt

```
Please update data.py to add feature X.

Requirements:
1. Read CLAUDE.md to understand architecture
2. Follow docstring conventions
3. Add unit tests to test_data.py
4. Run both unit and integration tests
5. Report if anything breaks
```

---

## Replication Checklist

### To replicate this structure in a new project:

See **SETUP_CHECKLIST.md** for detailed step-by-step instructions.

**Quick version:**

```bash
# 1. Create project structure
mkdir -p src/project_name tests scripts data/{sample,raw,processed} .claude/agents

# 2. Copy essential files
cp CLAUDE.md PROJECT_TEMPLATE.md SETUP_CHECKLIST.md new_project/
cp -r .claude/ new_project/
cp .pre-commit-config.yaml pytest.ini pyproject.toml .gitignore new_project/
cp scripts/*.sh new_project/scripts/
cp tests/conftest.py new_project/tests/

# 3. Update project-specific values
# - Replace "survival_framework" with "new_project_name"
# - Update CLAUDE.md with new project description
# - Update model names, data paths, etc.

# 4. Initialize
cd new_project/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

---

## Benefits of This Structure

### 1. **Claude Code Works Autonomously**

- CLAUDE.md provides complete context
- Agents handle testing automatically
- Pre-commit hooks prevent bad code
- Integration tests catch workflow bugs

### 2. **Consistent Quality**

- Automated formatting (black, isort)
- Automated linting (flake8, pylint)
- Type checking (mypy)
- Security checks (bandit)
- Documentation coverage (interrogate)

### 3. **Reliable Testing**

- Unit tests for fast feedback
- Integration tests for thorough validation
- Both must pass before commit
- CI/CD ready

### 4. **Easy Onboarding**

- New developers read CLAUDE.md
- Scripts automate setup
- Conventions are documented
- Examples are provided

### 5. **Reproducible**

- Template documents structure
- Checklist guides setup
- All configurations included
- Works across projects

---

## Customization Points

### For Different Project Types

**Machine Learning Projects:**
- Add `notebooks/` for Jupyter notebooks
- Add `configs/` for experiment configs
- Add `data/external/` for external datasets
- Update CLAUDE.md with ML-specific patterns

**Web Applications:**
- Add `app/` for application code
- Add `static/` and `templates/` for frontend
- Add `migrations/` for database migrations
- Update CLAUDE.md with API patterns

**Data Pipelines:**
- Add `pipelines/` for DAG definitions
- Add `dbt/` for dbt models
- Add `airflow/` for Airflow DAGs
- Update CLAUDE.md with pipeline patterns

---

## Maintenance

### Keep Updated

1. **CLAUDE.md** - Document major changes
2. **Tests** - Add tests for new features
3. **Agents** - Update agent instructions
4. **Documentation** - Keep guides current

### When Adding New Features

1. Update CLAUDE.md architecture section
2. Add unit tests
3. Add integration tests if workflow changes
4. Update docstrings
5. Run all checks

---

## Success Metrics

### This structure is successful if:

✅ **Claude Code can work autonomously** - Reads CLAUDE.md and knows what to do
✅ **Tests catch bugs early** - Both unit and integration tests required
✅ **Code quality is consistent** - Pre-commit hooks enforce standards
✅ **New projects start fast** - Template + checklist = quick setup
✅ **Team productivity high** - Less debugging, more building

---

## Example: Starting a New Project

```bash
# 1. Clone template structure
cp -r survival_framework_v1/ my_new_project/
cd my_new_project/

# 2. Clean project-specific artifacts
rm -rf artifacts/ models/ data/sample/*.csv

# 3. Update project name
find . -type f -exec sed -i '' 's/survival_framework/my_new_project/g' {} +

# 4. Update CLAUDE.md
vim CLAUDE.md  # Update project description, architecture, commands

# 5. Initialize environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install

# 6. Verify setup
pytest -m "not integration" -v
./scripts/run-checks.sh

# 7. Start coding!
# Claude Code now has full context from CLAUDE.md
```

---

## Resources

### Documentation Files
- **CLAUDE.md** - Main context for Claude Code
- **PROJECT_TEMPLATE.md** - This file
- **SETUP_CHECKLIST.md** - Step-by-step setup
- **TESTING.md** - Testing guide
- **CODE_QUALITY.md** - Quality standards
- **INTEGRATION_TESTING_GUIDE.md** - Why integration tests matter

### Configuration Files
- **.pre-commit-config.yaml** - Pre-commit hooks
- **pytest.ini** - Pytest configuration
- **pyproject.toml** - Tool configurations
- **.gitignore** - Git ignore rules

### Scripts
- **setup-pre-commit.sh** - Install pre-commit
- **run-checks.sh** - Run quality checks
- **format-code.sh** - Auto-format code
- **run-integration-tests.sh** - End-to-end tests

### Agents
- **.claude/agents/test-runner.md** - Unit test agent
- **.claude/agents/integration-tester.md** - Integration test agent

---

## Summary

**This project template provides:**

1. ✅ **Complete context for Claude Code** (CLAUDE.md)
2. ✅ **Automated testing** (unit + integration)
3. ✅ **Quality enforcement** (pre-commit hooks)
4. ✅ **Specialized agents** (test-runner, integration-tester)
5. ✅ **Comprehensive documentation** (guides for everything)
6. ✅ **Easy replication** (template + checklist)

**Result:** Automated, bug-free agentic coding experience! 🚀

**Next:** See SETUP_CHECKLIST.md for step-by-step replication guide.
