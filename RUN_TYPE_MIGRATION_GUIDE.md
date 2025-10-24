# Run Type Migration Guide

This guide documents the changes needed to support sample vs production runs with flexible input formats (CSV/pickle).

## Summary of Changes

### 1. New Directory Structure

```
data/
├── inputs/
│   ├── sample/              # Sample data for development
│   │   ├── *.csv
│   │   └── *.pkl
│   └── production/          # Full population data
│       ├── *.csv
│       └── *.pkl
└── outputs/
    ├── sample/              # All outputs from sample runs
    │   ├── predictions/
    │   ├── artifacts/
    │   ├── models/
    │   └── mlruns/
    └── production/          # All outputs from production runs
        ├── predictions/
        ├── artifacts/
        ├── models/
        └── mlruns/
```

### 2. Code Changes Made

#### data.py
- ✅ Added `RunType = Literal["sample", "production"]`
- ✅ Added `load_data()` function supporting CSV and pickle
- ✅ Automatic format detection based on file extension

#### utils.py
- ✅ Added `RunType` type
- ✅ Updated `versioned_name()` to include run_type prefix
- ✅ Added `get_output_paths()` for standardized path management

### 3. Code Changes Needed

#### train.py
Update `train_all_models()` signature:
```python
def train_all_models(
    file_path: str,
    run_type: RunType = "sample"
):
    # Use load_data() instead of pd.read_csv()
    df = load_data(file_path, run_type=run_type)

    # Get output paths based on run_type
    paths = get_output_paths(run_type)

    # Use paths["artifacts"], paths["models"], etc.
    ph_flags_path = os.path.join(paths["artifacts"], "ph_flags.csv")

    # Update model saving with run_type
    model_path = os.path.join(
        paths["models"],
        versioned_name(f"{name}.joblib", run_type=run_type)
    )
```

#### predict.py
Update `generate_predictions()` signature:
```python
def generate_predictions(
    file_path: str,
    run_type: RunType = "sample",
    time_horizons: List[int] = None,
):
    # Load data with run_type
    df = load_data(file_path, run_type=run_type)

    # Get paths for this run type
    paths = get_output_paths(run_type)

    # Load best model from correct location
    model_name, pipeline = load_best_model(
        artifacts_dir=paths["artifacts"],
        models_dir=paths["models"]
    )

    # Save to correct predictions directory
    output_path = os.path.join(
        paths["predictions"],
        f"survival_predictions_{run_type}_{timestamp}.csv"
    )
```

#### main.py
Update to accept run_type:
```python
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/inputs/sample/survival_inputs_sample2000.csv",
        help="Path to input file (CSV or pickle)"
    )
    parser.add_argument(
        "--run-type",
        choices=["sample", "production"],
        default="sample",
        help="Run type: sample (development) or production (full data)"
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"RUN TYPE: {args.run_type.upper()}")
    print(f"INPUT: {args.input}")
    print(f"{'='*60}\n")

    # Train models
    train_all_models(args.input, run_type=args.run_type)

    # Generate predictions
    generate_predictions(args.input, run_type=args.run_type)
```

### 4. Usage Examples

#### Sample Run (Development)
```bash
# CSV input
python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample

# Pickle input
python src/main.py --input data/inputs/sample/survival_data.pkl --run-type sample
```

#### Production Run
```bash
# CSV input
python src/main.py --input data/inputs/production/all_customers.csv --run-type production

# Pickle input
python src/main.py --input data/inputs/production/all_customers.pkl --run-type production
```

### 5. Output Organization

All outputs now include run_type in paths and filenames:

**Sample Outputs:**
- `data/outputs/sample/predictions/survival_predictions_sample_20251023_143052.csv`
- `data/outputs/sample/artifacts/model_metrics.csv`
- `data/outputs/sample/models/sample_cox_ph_20251023_143052.joblib`
- `data/outputs/sample/mlruns/...`

**Production Outputs:**
- `data/outputs/production/predictions/survival_predictions_production_20251023_143052.csv`
- `data/outputs/production/artifacts/model_metrics.csv`
- `data/outputs/production/models/production_cox_ph_20251023_143052.joblib`
- `data/outputs/production/mlruns/...`

### 6. Migration Steps

1. ✅ Create new directory structure
2. ✅ Move existing sample data to `data/inputs/sample/`
3. ✅ Move existing predictions to `data/outputs/sample/predictions/`
4. ✅ Update `data.py` with `load_data()` function
5. ✅ Update `utils.py` with path management functions
6. ⏳ Update `train.py` to use run_type
7. ⏳ Update `predict.py` to use run_type
8. ⏳ Update `main.py` to accept run_type argument
9. ⏳ Update `.gitignore` for new structure
10. ⏳ Update METHODOLOGY.md documentation
11. ⏳ Test sample run
12. ⏳ Test production run (mock data)

### 7. Benefits

- ✅ **Clear separation**: Sample and production runs never conflict
- ✅ **Flexible input**: Supports both CSV and pickle formats
- ✅ **Easy identification**: Run type visible in all file names and paths
- ✅ **Reproducibility**: Each run type maintains its own artifacts
- ✅ **Safety**: Production runs won't overwrite development work
- ✅ **Scalability**: Pickle format supports large datasets efficiently

### 8. Backward Compatibility

Old code will break. Migration required for:
- Direct references to `artifacts/`, `models/`, `mlruns/` directories
- `pd.read_csv()` calls should use `load_data()`
- Path construction should use `get_output_paths()`

### 9. Testing Checklist

- [ ] Sample CSV input loads correctly
- [ ] Sample pickle input loads correctly
- [ ] Production CSV input loads correctly
- [ ] Production pickle input loads correctly
- [ ] Sample outputs go to `data/outputs/sample/`
- [ ] Production outputs go to `data/outputs/production/`
- [ ] Filenames include run_type prefix
- [ ] Cross-validation works for both run types
- [ ] Predictions work for both run types
- [ ] MLflow tracking separates by run type
