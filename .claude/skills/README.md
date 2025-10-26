# Claude Code Skills

This directory contains Claude Code skills for the survival analysis framework.

## Available Skills

### unit-tests

Run and analyze unit tests (pytest) for the survival modeling framework.

**Usage:**
```
/skill unit-tests
```

**What it does:**
- Runs fast, isolated pytest unit tests
- Analyzes test failures
- Checks code quality (black, isort, flake8, mypy)
- Maintains >80% test coverage
- Provides TDD workflow guidance

**Focus**: Individual functions and modules in isolation

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

### logging-policy

Guide for implementing comprehensive runtime logging throughout the framework.

**Usage:**
```
/skill logging-policy
```

**What it does:**
- Defines multi-level logging strategy (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Sets up hierarchical logger organization
- Creates four log files: main, performance, warnings, debug
- Provides timing decorators and context managers
- Implements warning categorization system
- Guides performance metrics logging

**Use when:**
- Adding logging to new modules
- Need to track execution timing
- Want to categorize and analyze warnings
- Adding performance monitoring

**Focus**: Runtime execution monitoring and debugging

---

### context-management

Templates and best practices for maintaining high-quality context files (CLAUDE.md, README.md, CHANGELOG.md).

**Usage:**
```
/skill context-management
```

**What it does:**
- Provides templates for context file sections
- Documents pre-commit hook enforcement (automated verification)
- Guides best practices for writing effective context
- Includes `analyze_context.py` tool for quality analysis
- Scores context quality (0-100) with recommendations
- Checks git sync status

**Use when:**
- Writing or updating CLAUDE.md sections
- Need templates for context structure
- Want to analyze context quality
- Reference for what makes good context

**Note**: Context updates are now enforced by pre-commit hook (automatic), not manual workflow

---

### changelog-management

Guide for maintaining CHANGELOG.md with human-readable change history.

**Usage:**
```
/skill changelog-management
```

**What it does:**
- Follows Keep a Changelog format with semantic versioning
- Categorizes changes: Added, Changed, Deprecated, Removed, Fixed, Security
- Provides templates and best practices for changelog entries
- Guides version release workflow
- Includes automation script for generating entries from git history
- Integrates with post-push workflow

**Use when:**
- After every git push with significant changes
- Before version releases
- Adding new features, bug fixes, or breaking changes

**Trigger**: After git push, alongside context-management

**Focus**: Human-readable project history and troubleshooting

---

### archive-management

Automatically identify and archive completed task artifacts to keep the codebase clean.

**Usage:**
```
/skill archive-management
```
or user requests: "clean up old planning docs", "archive completed tasks"

**What it does:**
- Scans for old assessment files (>30 days), planning docs (>7 days), one-off scripts
- Presents candidates for review with age, size, and archival reason
- Archives to `.claude/archive/` organized by type and month
- Uses `git mv` to preserve file history
- Creates archive README documenting what was archived and why
- Commits changes with descriptive message

**Safety features:**
- 7-day safety buffer (never archives recent files)
- Confirmation required before moving files
- Dry run mode available
- Easy rollback with git

**Use when:**
- Noticing >5 old assessment files
- After major milestone completion
- Periodic maintenance (monthly)
- Before releases

**Proactive**: Claude will suggest running this when detecting accumulated old files

**Focus**: Codebase cleanliness while preserving historical context

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
