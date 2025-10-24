# Claude Code Skills

This directory contains Claude Code skills for the survival analysis framework.

## Available Skills

### test-suite

Run and analyze the test suite with pytest. Includes test execution, coverage analysis, and code quality checks.

**Usage:**
```
/skill test-suite
```

**What it does:**
- Runs pytest with coverage
- Analyzes test failures
- Checks code quality (black, isort, flake8, mypy)
- Maintains >80% test coverage
- Provides TDD workflow guidance

---

### integration-tests

Validate the complete end-to-end training pipeline with real data.

**Usage:**
```
/skill integration-tests
```

**What it does:**
- Runs full pipeline with sample data
- Verifies all models train successfully
- Checks artifacts are created correctly
- Validates metrics are in expected ranges
- Detects integration bugs early

---

### validate-model-outputs

Comprehensive validation of model predictions and artifacts.

**Usage:**
```
/skill validate-model-outputs
```

**What it does:**
- Validates survival function predictions
- Checks risk score distributions
- Verifies metrics are in valid ranges
- Validates model artifact files
- Ensures cross-model consistency
- Detects data quality issues

---

### parameter-management

Guide for centralizing and documenting all configurable parameters.

**Usage:**
```
/skill parameter-management
```

**What it does:**
- Provides patterns for parameter organization
- Shows how to document parameters with docstrings
- Demonstrates environment-specific configurations
- Guides when to add new parameters vs keep constants
- Includes examples for hyperparameters, data config, analysis config
- Shows JSON serialization for experiment tracking

**Use when:**
- Adding new hyperparameters to models
- Introducing configurable data features
- Creating analysis parameters (time horizons, CV folds)
- Need to document what a parameter controls

---

## Skill vs Agent

**Skills** are preferred over agents because they:
- Execute faster (single invocation)
- Provide focused functionality
- Are easier to compose
- Have clearer scope
- Better for interactive use

Use skills for:
- Running tests
- Validating outputs
- Executing specific workflows

Use the Task tool (agents) for:
- Complex multi-step exploration
- Researching unfamiliar codebases
- Tasks requiring multiple rounds of search

---

## Creating New Skills

Skills are markdown files in `.claude/skills/` that provide:
1. Clear purpose and usage instructions
2. Commands to execute
3. Validation criteria
4. Example usage
5. Expected results

Keep skills focused on a single workflow or validation task.
