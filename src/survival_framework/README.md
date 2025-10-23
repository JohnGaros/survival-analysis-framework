# survival_framework/

## Purpose

Main Python package containing survival modeling implementation.

## Modules

- `data.py` - Data loading, preprocessing pipeline, and feature engineering
- `models.py` - Survival model wrappers (CoxPH, Coxnet, WeibullAFT, GBSA, RSF)
- `metrics.py` - Evaluation metrics (C-index, IBS, Brier score)
- `validation.py` - Cross-validation logic and model evaluation
- `utils.py` - Utility functions (time grids, structured arrays)
- `train.py` - Main training pipeline orchestration

## Architecture

**Data Flow:**
1. Load CSV → `data.py::load_data()`
2. Preprocess → `data.py::make_preprocessor()` + `make_pipeline()`
3. Split folds → `validation.py::event_balanced_splitter()`
4. Train models → `validation.py::evaluate_model()` per fold
5. Save results → `train.py::train_all_models()`

## Key Design Patterns

- **Pipeline Pattern:** Preprocessing + VarianceThreshold + Model
- **Wrapper Pattern:** Consistent interface across different survival libraries
- **Dataclass Models:** Type-safe configuration and model definitions

See `CLAUDE.md` for complete architecture documentation.
