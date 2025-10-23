# src/

## Purpose

Source code directory containing all application code.

## Contents

- `survival_framework/` - Main Python package with survival modeling modules
- `main.py` - Entry point for training pipeline

## Structure

All production code lives in this directory. The package is organized by functionality:
- Data loading and preprocessing
- Model implementations
- Evaluation metrics
- Cross-validation logic
- Training pipeline orchestration

## Usage

```bash
# Run training pipeline
cd src/
python main.py

# Import package modules
from survival_framework.data import load_data, split_X_y
from survival_framework.models import build_models
```

**Note:** Add `src/` to PYTHONPATH when running tests or importing modules.
