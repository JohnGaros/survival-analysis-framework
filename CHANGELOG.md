# Changelog

All notable changes to the Survival Analysis Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Code reviewer subagent for automated code quality checks
  - Created `.claude/agents/code-reviewer.md` - specialized AI agent for code review
  - Automatically invoked after code changes to ensure quality and correctness
  - Survival analysis expertise (structured arrays, risk scores, IPCW, time grids)
  - Checks Python best practices, security, testing, documentation
  - Enforces project standards (config usage, MLflow tracking, context updates)
  - Provides prioritized feedback (Critical/Warnings/Suggestions)
  - Created `.claude/agents/README.md` with agent documentation

- Automated post-commit hook for GitHub synchronization
  - Created `.git/hooks/post-commit` to automatically push commits to GitHub
  - Detects current branch and remote tracking automatically
  - Graceful error handling with helpful messages
  - 60-second timeout to prevent hanging
  - Logs failures to `.git/hooks-logs/post-commit-failures.log`
  - Never blocks commits even if push fails

- Automated pre-commit hook for context file verification
  - Created `scripts/pre_commit_context_check.py` (258 lines) to enforce CHANGELOG.md and CLAUDE.md updates
  - Detects significant changes (src/*.py, requirements.txt, etc.) and verifies context files are updated
  - Blocks commits if CHANGELOG.md not updated for significant code changes
  - Warns if CLAUDE.md hasn't been updated in >7 days
  - Integrated into `.pre-commit-config.yaml` as `check-context-files` hook
  - Provides helpful error messages with update instructions

- Changelog management skill with Keep a Changelog format
  - Created `.claude/skills/changelog-management.md` with comprehensive guidelines
  - Integrated with post-push workflow alongside context management
  - Provides templates and best practices for human-readable change documentation

### Changed
- Refactored context-management skill from workflow enforcer to reference guide
  - Changed purpose from "trigger after git push" to "templates and best practices"
  - Removed redundant "When to Use This Skill" section (now handled by pre-commit hook)
  - Added "Quick Reference" section highlighting automated enforcement
  - Converted "After Git Push Checklist" to "Manual Context Update Checklist"
  - Updated `.claude/skills/README.md` to reflect new purpose
  - Skill now focuses on HOW to write good context, not WHEN (automated by hook)

## [0.3.0] - 2025-10-25

### Added
- Comprehensive logging system with multi-level log files
  - Real-time console output with INFO level messages
  - Four log files: main (detailed), performance (metrics only), warnings (issues only), debug (verbose)
  - Performance logging with `log_performance()` function and structured metrics
  - Warning categorization system (convergence, numerical, data, statistical)
  - Timer decorator `@log_execution_time()` for function-level timing
  - Timer context manager for code block timing
  - ProgressLogger class for tracking long-running operations
  - All logs stored in `data/outputs/{run_type}/logs/` with timestamps
  - Created `src/survival_framework/logging_config.py` (348 lines)
  - Created `src/survival_framework/timing.py` (151 lines)
  - Created `.claude/skills/logging-policy.md` (668 lines) with implementation guide

- Context management skill with automated quality analysis
  - Created `.claude/skills/context-management.md` (523 lines) with post-push workflow
  - Created `scripts/analyze_context.py` (501 lines) for context optimization
  - Analyzes CLAUDE.md, README.md, module READMEs, git sync status, and skills
  - Weighted scoring system (0-100) with letter grades
  - Priority-based recommendations (HIGH, MEDIUM, LOW)
  - Git sync detection comparing CLAUDE.md updates vs commit history
  - Visual progress bars in terminal output
  - JSON export for tracking context quality over time

- Skills organization and documentation
  - Renamed `test-suite.md` to `unit-tests.md` for clarity
  - Clear distinction between unit-tests (fast pytest) and integration-tests (E2E pipeline)
  - Updated `.claude/skills/README.md` with skill descriptions and focus areas

### Changed
- Parameterized hardcoded filenames in context files for better maintainability
  - CLAUDE.md now uses `<path_to_csv_file>` placeholder instead of hardcoded paths
  - Integration tests use `<sample_file>.csv` template with concrete example below
  - Updated directory structure documentation to match actual layout (`data/inputs/{sample,production}/`)

- Integrated logging throughout the pipeline
  - `src/main.py`: Added logging setup, Timer wrappers for training/prediction phases
  - `src/survival_framework/train.py`: Added per-model loggers, performance metrics, warning capture
  - All training steps now log timing, metrics, and progress

- Improved CLAUDE.md structure and accuracy
  - Updated directory tree to show `.claude/skills/` structure
  - Corrected data paths from legacy `data/sample/` to `data/inputs/sample/`
  - Added Recent Updates section with rolling last 5 changes
  - Added detailed logging system documentation

### Fixed
- Git rename issues when moving skill files (used proper `git mv` workflow)

## [0.2.0] - 2025-10-24

### Added
- Run-type-based organization for inputs and outputs
  - Separate `data/inputs/sample/` and `data/inputs/production/` directories
  - Separate `data/outputs/sample/` and `data/outputs/production/` with subdirs for models, artifacts, mlruns, logs
  - `--run-type` CLI parameter (sample or production)
  - Test outputs in dedicated `data/outputs/test_outputs/` directory

- Centralized configuration system
  - Created `src/survival_framework/config.py` with dataclasses for all hyperparameters
  - ModelConfig: Model-specific hyperparameters (n_estimators, learning_rate, etc.)
  - DataConfig: Feature lists, column names, imputation strategies
  - AnalysisConfig: Time horizons, CV folds, metrics
  - ExecutionConfig: Parallelization settings (sequential, mp, polars)
  - All parameters documented with docstrings
  - Single source of truth for tunable parameters

- Execution mode support
  - Sequential mode for debugging
  - Multiprocessing mode for parallel training
  - Polars mode for large dataset processing (future)
  - `--execution-mode` and `--n-jobs` CLI parameters

### Changed
- Migrated from legacy directory structure to run-type-based organization
  - Removed duplicate model storage in `src/models/` (moved to `data/outputs/{run_type}/models/`)
  - Removed duplicate artifacts in `src/artifacts/` (moved to `data/outputs/{run_type}/artifacts/`)
  - Freed 382MB of disk space by consolidating output directories

- Updated all module imports to use centralized config
  - `train.py`, `validation.py`, `models.py` now use config dataclasses
  - Removed hardcoded hyperparameters scattered across modules

### Removed
- Legacy directory structure (`src/models/`, `src/artifacts/`, `src/mlruns/`)
- 161 outdated markdown documentation files
  - Removed from git tracking to reduce repository clutter
  - Added `PHASE1_BENCHMARK.md` to .gitignore
  - Kept only actively maintained documentation (CLAUDE.md, README.md, CODE_QUALITY.md, TESTING.md)

### Fixed
- Production data preprocessing issues
  - Fixed OneHotEncoder handling of unknown categories (set `handle_unknown='ignore'`)
  - Added robust missing value imputation for production data edge cases

## [0.1.0] - 2025-10-23

### Added
- Initial survival analysis framework
  - Cox Proportional Hazards model
  - Elastic Net Cox (Coxnet)
  - Weibull AFT model
  - Gradient Boosting Survival Analysis (GBSA)
  - Random Survival Forest (RSF)

- Cross-validation with event-balanced stratification
  - 5-fold stratified CV ensuring proportional event distribution
  - Per-fold metrics: C-index, Integrated Brier Score, time-dependent AUC
  - Fold-wise survival predictions and risk scores saved to artifacts

- MLflow experiment tracking
  - All runs logged under "survival_framework" experiment
  - Per-fold metrics logged
  - Model artifacts attached to runs

- Proportional hazards assumption testing
  - Schoenfeld residuals test for Cox models
  - Results saved to `artifacts/ph_flags.csv`

- Comprehensive test suite
  - pytest-based testing with >80% coverage
  - Test fixtures for sample data
  - Tests for data loading, preprocessing, metrics, utilities
  - Pre-commit hooks for code quality (black, isort, flake8, mypy, pylint)

- Sample data for development
  - `data/inputs/sample/survival_inputs_sample2000.csv` (~1500 records)
  - All required features included for testing

### Fixed
- LAPACK numerical stability errors in Cox models
  - Added L2 regularization (alpha=0.01) to prevent singular Hessian matrices
  - Implemented comprehensive missing value imputation with missing indicators
  - Added variance threshold filtering (threshold=0.01) to remove constant features
  - Fold-aware time grid constraints for cross-validation to prevent extrapolation
  - See `LAPACK_FIX_SUMMARY.md` for detailed technical analysis

- Data preprocessing robustness
  - Handle edge cases with all-NaN columns
  - Proper scaling after imputation
  - Consistent feature ordering across CV folds

## References

- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

[Unreleased]: https://github.com/JohnGaros/survival-analysis-framework/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/JohnGaros/survival-analysis-framework/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/JohnGaros/survival-analysis-framework/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JohnGaros/survival-analysis-framework/releases/tag/v0.1.0
