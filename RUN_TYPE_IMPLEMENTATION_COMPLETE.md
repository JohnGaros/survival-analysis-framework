# Run Type Implementation - COMPLETE

## Summary

Successfully implemented a comprehensive system to distinguish between sample and production runs with flexible input/output formats.

## ✅ Completed Changes

### 1. Directory Structure
```
data/
├── inputs/
│   ├── sample/                  # Development data
│   │   ├── survival_inputs_sample2000.csv
│   │   └── README.md
│   └── production/              # Full population data (empty, ready for use)
└── outputs/
    ├── sample/                  # Sample run outputs
    │   ├── predictions/
    │   ├── artifacts/
    │   ├── models/
    │   └── mlruns/
    └── production/              # Production run outputs
        ├── predictions/
        ├── artifacts/
        ├── models/
        └── mlruns/
```

### 2. Code Updates

#### data.py ✅
- Added `RunType = Literal["sample", "production"]`
- Created `load_data(file_path, run_type)` supporting CSV and pickle
- Automatic format detection (.csv, .pkl, .pickle)
- Run type logging

#### utils.py ✅
- Added `RunType` type
- Updated `versioned_name(base, run_type)` to prefix filenames
- Created `get_output_paths(run_type)` for path management

#### train.py ✅
- Updated `train_all_models(file_path, run_type="sample")`
- Uses `load_data()` instead of `pd.read_csv()`
- Uses `get_output_paths()` for all outputs
- MLflow run names include run_type
- Model filenames prefixed with run_type
- Progress messages show run_type

#### predict.py ✅
- Updated `generate_predictions(file_path, run_type="sample")`
- Uses `load_data()` for input
- Uses `get_output_paths()` for outputs
- Prediction filenames include run_type
- Loads models from correct run_type directory

#### main.py ✅
- Complete CLI with argparse
- `--input` for file path
- `--run-type` for sample/production
- `--predict-only` flag
- `--train-only` flag
- Comprehensive usage examples
- Professional output formatting

### 3. Usage Examples

#### Default (Sample Run)
```bash
python src/main.py
# Uses default: data/inputs/sample/survival_inputs_sample2000.csv, run_type=sample
```

#### Sample Run with CSV
```bash
python src/main.py \
  --input data/inputs/sample/survival_inputs_sample2000.csv \
  --run-type sample
```

#### Production Run with Pickle
```bash
python src/main.py \
  --input data/inputs/production/all_customers.pkl \
  --run-type production
```

#### Predict Only (No Training)
```bash
python src/main.py \
  --input data/inputs/sample/data.csv \
  --run-type sample \
  --predict-only
```

#### Train Only (No Predictions)
```bash
python src/main.py \
  --input data/inputs/production/data.pkl \
  --run-type production \
  --train-only
```

### 4. Output Organization

**Sample Outputs:**
- Predictions: `data/outputs/sample/predictions/survival_predictions_sample_YYYYMMDD_HHMMSS.csv`
- Artifacts: `data/outputs/sample/artifacts/model_metrics.csv`
- Models: `data/outputs/sample/models/sample_cox_ph_YYYYMMDD_HHMMSS.joblib`
- MLflow: `data/outputs/sample/mlruns/...`

**Production Outputs:**
- Predictions: `data/outputs/production/predictions/survival_predictions_production_YYYYMMDD_HHMMSS.csv`
- Artifacts: `data/outputs/production/artifacts/model_metrics.csv`
- Models: `data/outputs/production/models/production_cox_ph_YYYYMMDD_HHMMSS.joblib`
- MLflow: `data/outputs/production/mlruns/...`

### 5. Key Features

✅ **Input Flexibility:**
- CSV format (human-readable)
- Pickle format (efficient for large datasets)
- Automatic format detection

✅ **Clear Separation:**
- Sample and production outputs never conflict
- Run type visible in all paths and filenames
- Each run type maintains its own artifacts

✅ **Safety:**
- Production runs won't overwrite development work
- Easy to identify which data was used
- Reproducible runs with clear provenance

✅ **Scalability:**
- Pickle supports large datasets efficiently
- Separate directories prevent file system clutter
- MLflow tracking separated by run type

### 6. Breaking Changes from Original

The following old code patterns will **not work**:

❌ **Old (won't work):**
```python
train_all_models("data/sample/survival_inputs_sample2000.csv")
generate_predictions(
    csv_path="data.csv",
    output_dir="data/predictions",
    artifacts_dir="artifacts",
    models_dir="models"
)
```

✅ **New (correct):**
```python
train_all_models(
    "data/inputs/sample/survival_inputs_sample2000.csv",
    run_type="sample"
)
generate_predictions(
    "data/inputs/sample/survival_inputs_sample2000.csv",
    run_type="sample"
)
```

### 7. Testing Checklist

Need to test:
- [ ] Sample CSV input
- [ ] Sample pickle input (create test file)
- [ ] Production CSV input (create test file)
- [ ] Production pickle input (create test file)
- [ ] --predict-only flag
- [ ] --train-only flag
- [ ] File paths are correct
- [ ] Filenames include run_type
- [ ] MLflow tracking works
- [ ] Cross-validation still works
- [ ] Predictions still work

### 8. Remaining Tasks

1. **Update .gitignore**
   - Add `data/outputs/*/` patterns

2. **Create README files**
   - `data/inputs/README.md`
   - `data/outputs/README.md`
   - `data/inputs/production/README.md`
   - Update `data/inputs/sample/README.md`

3. **Update METHODOLOGY.md**
   - Document new I/O structure
   - Add run_type explanation
   - Update examples

4. **Test end-to-end**
   - Run sample test
   - Verify outputs
   - Test pickle format

### 9. Benefits Achieved

✅ **Clarity:** Impossible to confuse sample vs production runs
✅ **Safety:** No accidental overwrites between run types
✅ **Flexibility:** Supports multiple input formats
✅ **Organization:** Clean separation of all outputs
✅ **Scalability:** Ready for large production datasets
✅ **Reproducibility:** Clear provenance for every run
✅ **Professional:** Enterprise-ready structure

### 10. Migration for Existing Users

If you have existing code using the old API:

1. Move input data to `data/inputs/sample/` or `data/inputs/production/`
2. Update function calls to include `run_type` parameter
3. Update paths to use `get_output_paths()`
4. Remove references to old `artifacts/`, `models/` directories

### 11. Next Steps

1. Test the implementation with sample data
2. Create README files for new directories
3. Update .gitignore
4. Update METHODOLOGY.md
5. Commit changes
6. Update GITHUB_SETUP.md if needed

## Files Modified

- ✅ `src/survival_framework/data.py`
- ✅ `src/survival_framework/utils.py`
- ✅ `src/survival_framework/train.py`
- ✅ `src/survival_framework/predict.py`
- ✅ `src/main.py`
- ⏳ `.gitignore` (pending)
- ⏳ Various README.md files (pending)
- ⏳ `METHODOLOGY.md` (pending)

## Command Line Help

```bash
$ python src/main.py --help

usage: main.py [-h] [--input INPUT] [--run-type {sample,production}]
               [--predict-only] [--train-only]

Survival Analysis Framework - Train models and generate predictions

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to input file (CSV or pickle). Default: sample CSV
  --run-type {sample,production}
                        Run type: 'sample' for development, 'production' for full data. Default: sample
  --predict-only        Skip training and only generate predictions using existing models
  --train-only          Only train models, skip prediction generation

Examples:
  # Sample run with CSV
  python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample

  # Production run with pickle
  python src/main.py --input data/inputs/production/all_customers.pkl --run-type production

  # Skip training, only predict
  python src/main.py --input data/inputs/sample/data.csv --run-type sample --predict-only
```

---

**Status:** Core implementation COMPLETE ✅
**Next:** Testing and documentation updates ⏳
