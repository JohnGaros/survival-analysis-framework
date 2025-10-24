# src/

## Purpose

Source code directory containing all application code for the survival analysis framework.

## Contents

### Core Files

- **`main.py`** - Main entry point for the training and prediction pipeline
  - CLI with argument parsing
  - Supports CSV and pickle input formats
  - Configurable execution modes (pandas, multiprocessing)
  - Configuration file support via `--config` argument

### Configuration Files

- **`configs/`** - Configuration management directory
  - `sample.json` - Default configuration for sample/development data
  - `production.json` - Optimized configuration for large production datasets
  - Custom configurations can be created and loaded via CLI

### Package: `survival_framework/`

Main Python package organized by functionality:

#### Core Modules

- **`config.py`** - Centralized parameter configuration system
  - `ExecutionConfig` - Execution mode and parallelization settings
  - `ModelHyperparameters` - All model hyperparameters with environment presets
  - `DataConfig` - Feature definitions and preprocessing options
  - `AnalysisConfig` - Cross-validation and evaluation settings
  - `SurvivalFrameworkConfig` - Master configuration with JSON save/load

- **`data.py`** - Data loading and preprocessing
  - `load_data()` - Load CSV or pickle files
  - `split_X_y()` - Extract features and create structured survival arrays
  - `make_preprocessor()` - Create sklearn preprocessing pipeline
  - `to_structured_y()` - Convert to scikit-survival format

- **`models.py`** - Survival model implementations
  - `BaseSurvivalModel` - Base class with unified interface
  - `CoxPHWrapper` - Cox Proportional Hazards (L2 regularization)
  - `CoxnetWrapper` - Elastic Net Cox (L1 + L2 regularization)
  - `StratifiedCoxWrapper` - Stratified Cox for non-proportional hazards
  - `WeibullAFTWrapper` - Weibull Accelerated Failure Time model
  - `GBSAWrapper` - Gradient Boosting Survival Analysis
  - `RSFWrapper` - Random Survival Forest
  - All wrappers support configurable hyperparameters

- **`train.py`** - Training pipeline orchestration
  - `build_models()` - Instantiate all models with hyperparameters
  - `train_all_models()` - Complete training workflow with cross-validation
  - Supports sequential (pandas) and parallel (multiprocessing) execution
  - Integration with configuration system

- **`validation.py`** - Cross-validation and model evaluation
  - `event_balanced_splitter()` - Stratified K-fold splits
  - `evaluate_model()` - Fit and evaluate single model
  - `ph_assumption_flags()` - Proportional hazards testing

- **`metrics.py`** - Survival-specific metrics
  - Harrell's C-index (concordance index)
  - Integrated Brier Score (IBS)
  - Time-dependent AUC

- **`utils.py`** - Utility functions
  - Directory management
  - Time grid generation
  - Model versioning
  - Path resolution

- **`tracking.py`** - MLflow experiment tracking
  - Automatic logging of parameters, metrics, and artifacts
  - Experiment organization by run type

- **`predict.py`** - Prediction generation
  - Load trained models and generate predictions on new data

## Directory Structure

```
src/
├── main.py                          # Entry point
├── configs/                         # Configuration files
│   ├── sample.json                  # Sample data config
│   └── production.json              # Production-optimized config
└── survival_framework/              # Main package
    ├── __init__.py
    ├── config.py                    # Configuration system
    ├── data.py                      # Data loading & preprocessing
    ├── models.py                    # Model wrappers
    ├── train.py                     # Training orchestration
    ├── validation.py                # Cross-validation
    ├── metrics.py                   # Survival metrics
    ├── utils.py                     # Utilities
    ├── tracking.py                  # MLflow integration
    └── predict.py                   # Prediction generation
```

**Note**: All generated outputs (models, artifacts, MLflow tracking) are saved to `data/outputs/{run_type}/` to maintain separation between sample and production runs. See [Output Organization](#output-organization) below.

## Usage

### Basic Training

```bash
# Default sample run
python src/main.py

# Specify input file and run type
python src/main.py --input data/inputs/sample/data.csv --run-type sample

# Production run with pickle file
python src/main.py --input data/inputs/production/data.pkl --run-type production
```

### Using Configuration Files

```bash
# Use production-optimized configuration
python src/main.py --input data.pkl --config configs/production.json

# Use sample configuration
python src/main.py --input data.csv --config configs/sample.json
```

### Advanced Options

```bash
# Force multiprocessing with 8 cores
python src/main.py --input data.pkl --execution-mode mp --n-jobs 8

# Silent execution
python src/main.py --input data.pkl --verbose 0

# Training only (skip predictions)
python src/main.py --input data.csv --train-only

# Prediction only (use existing models)
python src/main.py --input data.csv --predict-only
```

### Python API

```python
# Import modules
from survival_framework.data import load_data, split_X_y
from survival_framework.models import build_models
from survival_framework.config import SurvivalFrameworkConfig

# Load data
df = load_data("data/inputs/sample/data.csv")
X, y, ids = split_X_y(df)

# Build models with configuration
config = SurvivalFrameworkConfig.for_run_type("production")
models = build_models(config.hyperparameters)

# Train specific model
from sklearn.pipeline import Pipeline
from survival_framework.data import make_preprocessor

preprocessor = make_preprocessor(numeric=X.columns, categorical=[])
pipeline = Pipeline([("preprocess", preprocessor), ("model", models["gbsa"])])
pipeline.fit(X, y)
```

## Configuration System

The framework uses a centralized configuration system that supports:

1. **Environment-Specific Presets**: Optimized settings for sample/production/experiment
2. **JSON Serialization**: Save and load configurations for reproducibility
3. **Type Safety**: Dataclasses with validation
4. **Comprehensive Documentation**: Inline docstrings explain all parameters
5. **Backward Compatibility**: Existing code works without changes

### Creating Custom Configurations

```python
from survival_framework.config import SurvivalFrameworkConfig

# Start with production preset
config = SurvivalFrameworkConfig.for_run_type("production")

# Customize hyperparameters
config.hyperparameters.gbsa_n_estimators = 30  # Even faster
config.hyperparameters.rsf_n_estimators = 50

# Add description
config.description = "Custom fast configuration for experimentation"

# Save for reuse
config.save("configs/custom.json")

# Load later
config = SurvivalFrameworkConfig.load("configs/custom.json")
```

## Performance Optimization

The production configuration (`configs/production.json`) provides significant performance improvements for large datasets:

- **GBSA**: 50 estimators (vs 100), subsample=0.5, min_samples_split=100
  - Expected speedup: ~2× faster
- **RSF**: 100 estimators (vs 300), max_depth=10, min_samples_split=100
  - Expected speedup: ~3× faster
- **Multiprocessing**: Automatic parallel execution with all CPU cores

On 287K records, GBSA training time reduced from 6.3 hours → ~1.5 hours.

## Module Organization

The package follows a clear separation of concerns:

- **Data Layer** (`data.py`) - Loading and preprocessing
- **Model Layer** (`models.py`) - Model implementations with unified interface
- **Evaluation Layer** (`validation.py`, `metrics.py`) - Cross-validation and metrics
- **Training Layer** (`train.py`) - Orchestration and workflow
- **Configuration Layer** (`config.py`) - Parameter management
- **Utilities** (`utils.py`, `tracking.py`, `predict.py`) - Support functions

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=survival_framework --cov-report=html

# Run specific test file
pytest tests/test_data.py
```

## Output Organization

All training outputs are organized by run type under `data/outputs/{run_type}/`:

```
data/outputs/
├── sample/                          # Sample run outputs
│   ├── models/                      # Saved model files
│   │   └── <model>_YYYYMMDD_HHMMSS.joblib
│   ├── artifacts/                   # Training artifacts
│   │   ├── ph_flags.csv             # Proportional hazards tests
│   │   ├── model_metrics.csv        # Per-fold CV metrics
│   │   ├── model_summary.csv        # Aggregated rankings
│   │   └── <model>/                 # Per-model predictions
│   │       ├── <model>_fold0_surv.npy
│   │       └── <model>_fold0_risk.npy
│   ├── predictions/                 # Generated predictions
│   │   └── predictions_YYYYMMDD_HHMMSS.csv
│   └── mlruns/                      # MLflow tracking
└── production/                      # Production run outputs
    └── (same structure as sample/)
```

**Benefits of this organization:**
- ✅ Separates development and production outputs
- ✅ Prevents accidental mixing of results
- ✅ Easy to clean up sample runs without affecting production
- ✅ Follows data science best practices

**Note**: All output directories are gitignored. Only configuration files under `src/configs/` are version controlled.

## Notes

- Add `src/` to PYTHONPATH when running tests or importing modules
- All generated outputs (models, artifacts, MLflow) are saved to `data/outputs/{run_type}/`
- Configuration files in `src/configs/` are version controlled for reproducibility
- Legacy `src/artifacts/`, `src/models/`, `src/mlruns/` directories are no longer used
