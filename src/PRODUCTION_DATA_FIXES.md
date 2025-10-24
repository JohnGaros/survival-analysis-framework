# Production Data Fixes - 2025-10-24

## Issues Found

### 1. Invalid Survival Times
**Error**: `ValueError: observed time contains values smaller zero`

**Root Cause**: The production dataset (`survival_inputs_complete.pkl`) contained 452 records with `survival_months ≤ 0`, which is invalid for survival analysis.

**Statistics**:
- Total records: 287,465
- Invalid records (survival_months ≤ 0): 452
- Minimum value: -7 months
- Valid records after filtering: 287,013

**Fix**: Updated `load_data()` function in `data.py` to automatically filter out records with survival_months ≤ 0:

```python
# Filter out invalid survival times (must be > 0)
if TIME_COL in df.columns:
    invalid_count = (df[TIME_COL] <= 0).sum()
    if invalid_count > 0:
        print(f"WARNING: Removing {invalid_count:,} records with survival_months ≤ 0")
        df = df[df[TIME_COL] > 0].copy()
        print(f"Remaining: {len(df):,} records")
```

### 2. Schema Mismatch - Optional Column
**Issue**: Production data contains `kwh_exp_smooth` column that is not present in sample data

**Columns in Production**:
- All 9 base numeric features
- **kwh_exp_smooth** (extra column)
- 2 categorical features

**Columns in Sample**:
- All 9 base numeric features (no kwh_exp_smooth)
- 2 categorical features

**Fix**: Made the framework flexible to handle optional columns:

1. Updated `NUM_COLS` to include `kwh_exp_smooth`
2. Modified `split_X_y()` to detect which columns are present:
```python
# Use only numeric columns that are actually present in the data
num_cols_present = [col for col in NUM_COLS if col in df.columns]
```

3. Updated `train_all_models()` to pass detected columns to preprocessor:
```python
# Detect which numeric columns are present in this dataset
num_cols_present = [col for col in NUM_COLS if col in X.columns]
pre = make_preprocessor(numeric=num_cols_present, categorical=CAT_COLS)
```

## Files Modified

1. **src/survival_framework/data.py**:
   - Added `kwh_exp_smooth` to NUM_COLS
   - Added invalid time filtering in `load_data()`
   - Made `split_X_y()` detect present columns

2. **src/survival_framework/train.py**:
   - Import NUM_COLS
   - Detect present columns before creating preprocessor

## Verification

Tested with production data (287K records):
- ✅ Invalid times filtered: 452 records removed
- ✅ Valid data loaded: 287,013 records
- ✅ Schema flexible: kwh_exp_smooth detected and included
- ✅ Preprocessing successful: 12 features → 26 transformed features
- ✅ No errors during data loading and preprocessing

## Benefits

1. **Robustness**: Framework now handles data quality issues automatically
2. **Flexibility**: Works with both sample (no kwh_exp_smooth) and production (with kwh_exp_smooth) data
3. **Transparency**: Warns user about filtered records
4. **Backward Compatibility**: Sample runs still work exactly as before

## Next Steps

The production pipeline is ready to run. Due to the large dataset size (287K records):
- Training will take significant time (5 models × 5 folds = 25 training runs)
- Consider running overnight or on a more powerful machine
- Predictions will be generated after training completes

**Command to run**:
```bash
python src/main.py --input data/inputs/production/survival_inputs_complete.pkl --run-type production
```

**Or programmatically**:
```python
from main import run_pipeline
run_pipeline(
    input_file="data/inputs/production/survival_inputs_complete.pkl",
    run_type="production"
)
```
