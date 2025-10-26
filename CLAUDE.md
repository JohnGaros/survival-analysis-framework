# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a survival analysis framework for modeling customer churn/termination using multiple survival models. The framework trains and evaluates Cox proportional hazards models, Weibull AFT, and Gradient Boosting survival models on financial customer data.

**For theoretical foundations and implementation details, see [`METHODOLOGY.md`](METHODOLOGY.md).**

## Recent Updates

**[2025-10-26]**: Stratum-level aggregate statistics for predictions
- **Added**: `strata_analysis.py` module for computing aggregate statistics by categorical strata
  - `create_stratum_identifier()` - Create composite stratum labels from categorical columns
  - `compute_strata_summary()` - Descriptive statistics (counts, event rates, survival times) by stratum
  - `compute_strata_predictions()` - Aggregate predictions (risk scores, survival probabilities) by stratum per fold
  - `compute_strata_metrics()` - Performance metrics (C-index, IBS, AUC) by stratum across folds
  - `save_strata_artifacts()` - Save stratum summaries to CSV
  - `aggregate_strata_predictions()` - Combine per-fold strata predictions into summary
- **Changed**: `validation.py::evaluate_model()` extended to compute and save per-fold stratum predictions
  - Now accepts `X_full`, `y_full`, `train_indices`, `test_indices`, `strata_cols` parameters
  - Automatically saves `{model}_fold{i}_strata.csv` files with aggregated predictions by stratum
  - Returns extended result dict with indices, risk scores, and times for downstream analysis
- **Changed**: `train.py::train_all_models()` integrated stratum analysis into pipeline
  - Computes and saves `strata_summary.csv` with baseline statistics before training
  - Passes strata columns to all fold evaluations for automatic aggregation
  - Modified `_train_single_fold()` to accept `strata_cols` parameter
- **Added**: `tests/test_strata_analysis.py` with comprehensive unit tests
- **Impact**: Enables identification of model performance heterogeneity across customer segments
  - Artifacts include: `strata_summary.csv`, `{model}/model_fold{i}_strata.csv`
  - Supports targeted interventions and segment-specific risk analysis
  - Facilitates stratified reporting and bias detection
- **New patterns**:
  - Strata defined by `CAT_COLS` (typeoftariff_coarse, risk_level_coarse)
  - Composite stratum identifiers use pipe separator: "tariff_A|risk_low"
  - Per-fold predictions aggregated by stratum with mean/std statistics
  - All stratum artifacts saved alongside existing model outputs
- **Files**: See `src/survival_framework/strata_analysis.py` and updated `validation.py`, `train.py`

**[2025-10-26]**: Automated code review and quality infrastructure
- **Added**: Code-reviewer subagent (`.claude/agents/code-reviewer.md`) for automated code quality checks
  - Specialized in survival analysis (structured arrays, risk scores, IPCW, time grids)
  - Enforces Python best practices, security, testing coverage (>80%)
  - Checks project standards (config.py usage, MLflow tracking, context updates)
  - Provides prioritized feedback: 🔴 Critical, 🟡 Warnings, 🟢 Suggestions
  - Automatically saves assessments to `.claude/assessments/code-review-YYYY-MM-DD.md`
- **Added**: Automated pre-commit hook (`scripts/pre_commit_context_check.py`)
  - Verifies CHANGELOG.md updated for significant changes (src/*.py, requirements.txt, .claude/, scripts/)
  - Warns if CLAUDE.md stale (>7 days since last update)
  - Blocks commits if context files not updated
  - Integrated into `.pre-commit-config.yaml`
- **Added**: Automated post-commit hook (`.git/hooks/post-commit`)
  - Automatically pushes commits to GitHub after each commit
  - Auto-detects current branch and remote tracking
  - Graceful error handling with 60-second timeout
  - Logs failures to `.git/hooks-logs/post-commit-failures.log`
- **Added**: Changelog management skill (`.claude/skills/changelog-management.md`)
  - Keep a Changelog format with semantic versioning
  - Integrated with post-push workflow
- **Changed**: Context-management skill refactored from enforcer to reference guide
  - Now provides templates and best practices for writing good context
  - Enforcement automated by pre-commit hook
- **Impact**: Fully automated git workflow with quality gates
  - Code changes → auto code review → verify context updated → commit → auto push
  - All reviews documented and tracked over time
  - Consistent code quality enforcement
- **New patterns**:
  - Code-reviewer runs proactively after Python code changes
  - Pre-commit hook blocks commits missing context updates
  - All assessments tracked in `.claude/assessments/` with history table
- **Files**: See `.claude/agents/README.md` and `.claude/assessments/README.md`

**[2025-10-25]**: Comprehensive logging system and context management
- **Added**: `logging_config.py` (348 lines) with multi-level logging (DEBUG, INFO, WARNING, ERROR)
- **Added**: `timing.py` (151 lines) with @log_execution_time() decorator and Timer context manager
- **Added**: `.claude/skills/logging-policy.md` (668 lines) - comprehensive logging guide
- **Added**: `.claude/skills/context-management.md` (523 lines) - context optimization framework
- **Added**: `scripts/analyze_context.py` (501 lines) - automated context quality analysis
- **Changed**: `main.py` and `train.py` now include extensive logging with performance tracking
- **Impact**: All runs now generate detailed logs in `data/outputs/{run_type}/logs/`
  - `main_{timestamp}.log` - Full execution log
  - `performance_{timestamp}.log` - Clean metrics (duration, cindex, ibs)
  - `warnings_{timestamp}.log` - Categorized warnings only
- **New patterns**:
  - Use `@log_execution_time()` decorator for timing functions
  - Use `Timer(logger, "description")` context manager for code blocks
  - Use `log_performance(logger, msg, **metrics)` for performance metrics
  - Use `capture_warnings(logger)` to categorize and log warnings
  - Run `python scripts/analyze_context.py` after git pushes to check context health
- **Files**: See `.claude/skills/logging-policy.md` and `.claude/skills/context-management.md`

**[2025-10-25]**: Centralized configuration system and markdown cleanup
- **Added**: `src/survival_framework/config.py` with dataclass-based configuration
  - `ModelHyperparameters` - All model parameters with environment presets
  - `DataConfig` - Feature definitions and preprocessing options
  - `AnalysisConfig` - Cross-validation settings
  - `SurvivalFrameworkConfig` - Master config with JSON save/load
- **Added**: `src/configs/sample.json` and `src/configs/production.json`
- **Changed**: Production config optimizes GBSA (50 estimators)
- **Removed**: 161 legacy files from git tracking (artifacts/, models/, mlruns/ at repo root)
- **Added**: `PHASE1_BENCHMARK.md` to .gitignore (development file)
- **Impact**:
  - Can now run: `python src/main.py --config configs/production.json`
  - Expected GBSA speedup: 6.3h → ~1.5h on production data
  - All outputs organized under `data/outputs/{run_type}/`
- **New patterns**: Use `SurvivalFrameworkConfig.for_run_type("production")` for optimized settings

**[2025-10-24]**: Run type system and production data migration
- **Added**: Run type system separating sample and production outputs
- **Changed**: All outputs now go to `data/outputs/{run_type}/` (models/, artifacts/, mlruns/, predictions/)
- **Fixed**: Production data issues (removed 452 invalid records with survival_months ≤ 0)
- **Impact**: Clean separation of development and production runs

**[2025-10-23]**: Prediction generation and LAPACK fixes
- **Added**: Prediction generation with survival probabilities at 3, 6, 12, 18, 24, 36 months
- **Added**: Expected survival time calculation (Restricted Mean Survival Time)
- **Fixed**: LAPACK numerical stability errors with L2 regularization
- **Added**: Comprehensive missing value imputation with indicators
- **Added**: Variance threshold filtering to remove constant features
- **Added**: Fold-aware time grid constraints for cross-validation
- See `LAPACK_FIX_SUMMARY.md` for detailed changes

## Running Commands

### Setup and Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Training Models

The main entry point is `src/main.py`, which calls `train_all_models()`:

```bash
python src/main.py
```

Or use the training module directly with custom data:

```bash
python -m survival_framework.train --csv <path_to_csv_file>
```

### Testing with Sample Data

A sample dataset is provided in `data/inputs/sample/` for testing and development. The sample file includes all required columns and can be used to verify the framework setup.

```bash
# Example with sample data
python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample
```

### Running Tests

The framework uses pytest for testing with comprehensive test coverage:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage

# Run specific test file
pytest tests/test_data.py

# Run fast tests only (skip slow tests)
pytest -m "not slow"

# Run tests in parallel for speed
pytest -n auto
```

### Test-Driven Development

Use the test-runner agent for TDD workflow:

1. **Write failing test first** - Test captures desired behavior
2. **Implement minimal code** - Make the test pass
3. **Refactor** - Clean up while keeping tests passing
4. **Check coverage** - Maintain >80% coverage

Test outputs are stored in `data/test_outputs/`:
- Coverage reports in `data/test_outputs/coverage/`
- Test artifacts in `data/test_outputs/artifacts/`
- Test models in `data/test_outputs/models/`

### Code Quality and Pre-commit Hooks

The framework uses pre-commit hooks to automatically check code quality:

```bash
# Setup pre-commit hooks (one-time)
./scripts/setup-pre-commit.sh

# Or manually
pip install -r requirements-dev.txt
pre-commit install
```

**Hooks run automatically on `git commit`:**
- **Context verification** (`scripts/pre_commit_context_check.py`) - Blocks commit if CHANGELOG.md not updated for significant changes
- Black (code formatting)
- isort (import sorting)
- Flake8 (linting)
- Mypy (type checking)
- Pylint (static analysis)
- Bandit (security checks)
- Interrogate (docstring coverage >80%)

**Post-commit automation:**
- `.git/hooks/post-commit` automatically pushes commits to GitHub after successful commit

**Manual execution:**
```bash
# Run all quality checks
./scripts/run-checks.sh

# Auto-format code
./scripts/format-code.sh

# Run pre-commit manually
pre-commit run --all-files
```

See `CODE_QUALITY.md` for detailed documentation.

## Architecture

### Directory Structure

```
survival_framework/
├── .claude/
│   ├── agents/                     # Specialized AI subagents
│   │   ├── README.md              # Agent documentation
│   │   ├── code-reviewer.md       # Automated code review agent
│   │   └── test-runner.md         # Test execution agent
│   ├── assessments/                # Code review assessments
│   │   ├── README.md              # Assessment documentation
│   │   └── code-review-YYYY-MM-DD.md  # Dated review reports
│   └── skills/                     # Reusable workflow skills
│       ├── README.md
│       ├── unit-tests.md
│       ├── integration-tests.md
│       ├── logging-policy.md
│       ├── context-management.md
│       ├── changelog-management.md
│       └── ...
├── data/
│   ├── inputs/                     # Input data files
│   │   ├── sample/                 # Sample datasets
│   │   │   ├── README.md
│   │   │   └── *.csv               # Sample CSV files
│   │   └── production/             # Production datasets
│   │       └── *.pkl               # Production pickle files
│   └── outputs/                    # All outputs (gitignored)
│       ├── sample/                 # Sample run outputs
│       │   ├── models/             # Trained models
│       │   ├── artifacts/          # Predictions and metrics
│       │   │   ├── strata_summary.csv          # Baseline statistics by stratum
│       │   │   ├── ph_flags.csv                # Proportional hazards test results
│       │   │   ├── model_metrics.csv           # Per-fold metrics
│       │   │   ├── model_summary.csv           # Aggregated model rankings
│       │   │   └── <model>/                    # Per-model directories
│       │   │       ├── <model>_fold{i}_surv.npy       # Survival predictions
│       │   │       ├── <model>_fold{i}_risk.npy       # Risk scores
│       │   │       └── <model>_fold{i}_strata.csv     # Stratum-aggregated predictions
│       │   ├── mlruns/             # MLflow tracking
│       │   └── logs/               # Execution logs
│       ├── production/             # Production run outputs
│       │   └── ...                 # Same structure as sample
│       └── test_outputs/           # Test run outputs
│           ├── coverage/           # Coverage reports
│           ├── artifacts/          # Test artifacts
│           └── models/             # Test models
├── src/
│   ├── main.py                     # Main entry point
│   ├── artifacts/                  # Generated during training (gitignored)
│   │   ├── ph_flags.csv           # Proportional hazards test results
│   │   ├── model_metrics.csv      # Per-fold CV metrics
│   │   ├── model_summary.csv      # Aggregated model rankings
│   │   └── <model>/               # Per-model directories
│   │       ├── <model>_fold0_surv.npy
│   │       └── <model>_fold0_risk.npy
│   ├── models/                     # Saved model pipelines (gitignored)
│   ├── mlruns/                     # MLflow experiment tracking (gitignored)
│   └── survival_framework/         # Main package
│       ├── __init__.py
│       ├── data.py                # Data loading and preprocessing
│       ├── models.py              # Model wrappers
│       ├── metrics.py             # Survival metrics
│       ├── validation.py          # Cross-validation utilities
│       ├── tracking.py            # MLflow integration
│       ├── utils.py               # Helper utilities
│       └── train.py               # Training orchestration
├── scripts/                        # Utility scripts
│   ├── setup-pre-commit.sh        # Setup pre-commit hooks
│   ├── run-checks.sh              # Run all quality checks
│   ├── format-code.sh             # Auto-format code
│   ├── pre_commit_context_check.py  # Pre-commit context verification
│   └── analyze_context.py         # Context quality analysis
├── .git/hooks/                     # Git hooks (not in version control)
│   ├── post-commit                # Auto-push to GitHub
│   └── ...                        # Other git hooks
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures and configuration
│   ├── test_data.py               # Tests for data module
│   ├── test_metrics.py            # Tests for metrics module
│   └── test_utils.py              # Tests for utils module
├── .gitignore
├── .pre-commit-config.yaml         # Pre-commit hooks configuration
├── pyproject.toml                  # Project and tool configuration
├── pytest.ini                      # Pytest configuration
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── readme.md
├── CLAUDE.md                       # This file
├── TESTING.md                      # Testing guide
└── CODE_QUALITY.md                 # Code quality guide
```

### Module Structure

The codebase follows a modular architecture under `src/survival_framework/`:

- **data.py**: Data loading, preprocessing, and pipeline construction

  - Defines expected column names (NUM_COLS, CAT_COLS, ID_COL, TIME_COL, EVENT_COL)
  - `split_X_y()`: Extracts features and creates structured survival arrays
  - `make_preprocessor()`: Creates sklearn ColumnTransformer with StandardScaler for numeric and OneHotEncoder for categorical features
  - `to_structured_y()`: Converts DataFrame to scikit-survival structured array with `('event', 'time')` fields

- **models.py**: Survival model wrappers with unified interface

  - All models inherit from `BaseSurvivalModel` with methods: `fit()`, `predict_survival_function()`, `score()`
  - Implements wrappers for: CoxPH, Coxnet (elastic net regularized Cox), Stratified Cox, Weibull AFT, GBSA
  - `StratifiedCoxWrapper` uses lifelines and requires original categorical columns in the DataFrame (not transformed)
  - `DeepSurvWrapper` is a placeholder for future torch/pycox integration

- **validation.py**: Cross-validation and model evaluation

  - `event_balanced_splitter()`: Creates stratified K-fold splits balanced on event indicator
  - `evaluate_model()`: Fits model, computes metrics (C-index, IBS, time-dependent AUC), saves fold predictions and stratum aggregates
  - `ph_assumption_flags()`: Runs Schoenfeld tests to check proportional hazards assumptions

- **strata_analysis.py**: Stratum-level aggregate statistics

  - `create_stratum_identifier()`: Combines categorical columns into composite stratum labels
  - `compute_strata_summary()`: Descriptive statistics (counts, event rates, survival times) by stratum
  - `compute_strata_predictions()`: Aggregates predictions (risk scores, survival probabilities) by stratum per fold
  - `compute_strata_metrics()`: Performance metrics (C-index, IBS, AUC) by stratum across folds
  - `save_strata_artifacts()`: Saves stratum summaries to CSV
  - `aggregate_strata_predictions()`: Combines per-fold strata predictions into summary

- **metrics.py**: Survival-specific metrics

  - Harrell's C-index via IPCW (inverse probability of censoring weighting)
  - Integrated Brier Score (IBS)
  - Time-dependent AUC using cumulative dynamic AUC

- **tracking.py**: MLflow integration for experiment tracking

  - All runs are logged under experiment name "survival_framework"

- **utils.py**: Helper functions for directory management, time grids, and model versioning

- **train.py**: Main training orchestration
  - `train_all_models()`: Trains all models with 5-fold cross-validation, logs to MLflow, saves artifacts

### Data Requirements

Input CSV must contain:

- **Numeric features**: debit_exp_smooth, credit_exp_smooth, balance_exp_smooth, past_due_balance_exp_smooth, oldest_past_due_exp_smooth, waobd_exp_smooth, total_settlements, active_settlements, defaulted_settlements
- **Categorical features**: typeoftariff_coarse, risk_level_coarse
- **ID column**: account_entities_key
- **Time column**: survival_months
- **Event column**: is_terminated (boolean or 0/1)

### Training Pipeline Flow

1. Load CSV and split into X, y (structured array), and IDs
2. Run proportional hazards assumption checks via Schoenfeld tests
3. Create preprocessing pipeline (StandardScaler + OneHotEncoder)
4. For each model:
   - Perform 5-fold event-balanced cross-validation
   - Evaluate on hold-out fold: compute C-index, IBS, time-dependent AUC
   - Save per-fold survival predictions and risk scores to `artifacts/<model>/`
   - Fit final model on full dataset and save to `models/` as versioned joblib
5. Aggregate cross-validation metrics and rank models

### Output Artifacts

- `artifacts/strata_summary.csv`: Baseline statistics by stratum (n_samples, event_rate, mean survival time)
- `artifacts/ph_flags.csv`: Proportional hazards test results per covariate
- `artifacts/<model>/<model>_fold{i}_surv.npy`: Survival function predictions per fold (n_samples × n_times)
- `artifacts/<model>/<model>_fold{i}_risk.npy`: Risk scores per fold (n_samples,)
- `artifacts/<model>/<model>_fold{i}_strata.csv`: Stratum-aggregated predictions per fold (mean risk, mean survival probabilities by time)
- `artifacts/model_metrics.csv`: Per-fold metrics for all models
- `artifacts/model_summary.csv`: Average metrics and rankings across folds
- `models/<model>_YYYYMMDD_HHMMSS.joblib`: Versioned fitted pipelines

### MLflow Tracking

All training runs are tracked in `src/mlruns/` under the experiment "survival_framework". Metrics are logged per fold, and artifacts (PH flags, fitted models) are attached to each run.

### Important Implementation Notes

- **StratifiedCoxWrapper** expects raw DataFrames with categorical columns intact (not one-hot encoded). This is handled by passing the original X DataFrame rather than the transformed array.
- **Lifelines models** (StratifiedCoxWrapper, WeibullAFTWrapper) require DataFrames with 'time' and 'event' columns appended.
- **scikit-survival models** (CoxPH, Coxnet, GBSA) work with numpy arrays and structured y.
- Risk scores convention: higher risk score = higher hazard = lower survival probability.
- The framework uses event-balanced stratified K-fold to ensure both censored and event samples are proportionally distributed across folds.

## Development Standards

- Python 3.11+
- Type hints required for all function signatures
- Test coverage >80%
- Follow PEP 8 style guidelines

### Methodology Documentation

**IMPORTANT**: When making code changes that affect the theoretical approach or implementation architecture, update `METHODOLOGY.md`:

- **Add new models**: Document theory, assumptions, and implementation details
- **Modify preprocessing**: Update Data Preprocessing Pipeline section
- **Change evaluation metrics**: Update Evaluation Metrics section
- **Alter prediction outputs**: Update Prediction Methodology section
- **Architectural changes**: Update Implementation Architecture section

The methodology document serves as the authoritative reference for understanding the framework's theoretical foundations and should always reflect the current implementation.

**DOCX Synchronization**: The framework automatically maintains `METHODOLOGY.docx` (Word format) in sync with `METHODOLOGY.md`:

- **Automatic**: Pre-commit hook auto-generates DOCX when MD changes
- **Manual**: Run `./scripts/sync-methodology.sh` to regenerate DOCX
- Both files are tracked in git for easy distribution

### Docstring Requirements

All functions, classes, and methods must include docstrings following Google style format:

```python
def function_name(param1: Type1, param2: Type2) -> ReturnType:
    """Brief one-line description.

    Extended description if needed, explaining the purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised

    Example:
        >>> function_name(value1, value2)
        expected_result
    """
```

**Key docstring guidelines:**
- Brief summary on first line (imperative mood: "Calculate...", "Return...", "Create...")
- Blank line before extended description
- Use `Args:`, `Returns:`, `Raises:`, `Yields:` (for generators), `Attributes:` (for classes)
- For survival-specific functions, document array shapes and dtypes (e.g., "y_struct: np.ndarray with dtype=[('event', bool), ('time', float)]")
- Document expected column names for DataFrame parameters
- Include Examples section for complex functions
- We are doing test-driven development and should use the sample data in the csv file for this purpose. Output data should be stored in an appropriate sub-directory of data. Create a sub-agent for this purpose along with all necessary artifacts