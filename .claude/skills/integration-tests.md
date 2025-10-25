# Integration Tests Skill

Validate the complete end-to-end training pipeline to ensure all components work together correctly.

## Purpose

This skill catches integration issues that unit tests miss by running the full pipeline with real data.

## When to Use

Run this skill:
1. After any code changes to data.py, models.py, validation.py, train.py
2. Before committing changes to ensure nothing breaks
3. After adding new models to verify pipeline compatibility
4. When changing preprocessing steps to verify downstream effects

## Testing Protocol

### Step 1: Clean Environment
```bash
rm -rf data/outputs/sample/artifacts/ data/outputs/sample/models/
```

### Step 2: Run End-to-End Pipeline
```bash
# Using sample data (adjust filename as needed)
python src/main.py --input data/inputs/sample/<sample_file>.csv --run-type sample

# Example with current sample file:
python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample
```

### Step 3: Verify Success

**Pipeline Execution:**
- ✅ All models train successfully
- ✅ No LAPACK errors
- ✅ No ValueError exceptions
- ✅ No IndexError exceptions
- ✅ Exit code 0

**Artifacts Created:**
- ✅ `data/outputs/sample/artifacts/model_metrics.csv`
- ✅ `data/outputs/sample/artifacts/model_summary.csv`
- ✅ `data/outputs/sample/artifacts/ph_flags.csv`
- ✅ `data/outputs/sample/models/*.joblib` (5 model files)

**Model Training Verification:**
- ✅ cox_ph: 5 folds completed
- ✅ coxnet: 5 folds completed
- ✅ weibull_aft: 5 folds completed
- ✅ gbsa: 5 folds completed
- ✅ rsf: 5 folds completed

**Performance Sanity Checks:**
- ✅ C-index > 0.5 (better than random)
- ✅ C-index < 1.0 (not overfitting)
- ✅ IBS > 0.0 (valid calibration)
- ✅ IBS < 0.25 (reasonable calibration)
- ✅ No NaN in metrics

## Validation Script

```python
import pandas as pd
import os

# Check artifacts exist
assert os.path.exists('data/outputs/sample/artifacts/model_metrics.csv')
assert os.path.exists('data/outputs/sample/artifacts/model_summary.csv')
assert os.path.exists('data/outputs/sample/artifacts/ph_flags.csv')

# Load metrics
metrics = pd.read_csv('data/outputs/sample/artifacts/model_metrics.csv')

# Check all models present
expected_models = {'cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf'}
actual_models = set(metrics['model'].unique())

if not expected_models.issubset(actual_models):
    raise ValueError(f"Missing models: {expected_models - actual_models}")

# Check metrics ranges
assert metrics['cindex'].between(0.5, 1.0).all(), "C-index out of range"
assert metrics['ibs'].between(0.0, 0.25).all(), "IBS out of range"
assert not metrics['cindex'].isna().any(), "NaN in C-index"
assert not metrics['ibs'].isna().any(), "NaN in IBS"

print("✅ All integration tests passed")
```

## Error Detection

### Critical Errors (Must Fix Immediately)

1. **LAPACK Errors** - Numerical instability
   ```
   ValueError: LAPACK reported an illegal value
   ```

2. **Time Range Errors** - Extrapolation issues
   ```
   ValueError: all times must be within follow-up time
   ```

3. **IndexErrors** - Array/DataFrame incompatibility
   ```
   IndexError: only integers, slices (...) are valid indices
   ```

4. **TypeErrors** - Type mismatches
   ```
   TypeError: DataFrame expected, got ndarray
   ```

### Warnings (Monitor But May Be Acceptable)

1. **Convergence Warnings** - Data quality issues
   ```
   ConvergenceWarning: Column X have very low variance
   ```

2. **Overflow Warnings** - Numerical but handled
   ```
   RuntimeWarning: overflow encountered in exp
   ```

## What It Tests

### 1. Preprocessing Pipeline Validation

Tests that preprocessing handles edge cases:
- Missing values in numeric features
- Missing values in categorical features
- Zero variance features
- Zero survival times
- Rare categorical levels

### 2. Cross-Validation Fold Compatibility

Ensures each fold has:
- Time points within test set range
- Sufficient events in each fold
- No empty predictions

### 3. Model Compatibility

Verifies all models:
- Accept preprocessed data format
- Complete all CV folds
- Produce valid predictions
- Save artifacts correctly

## Reporting

After running tests, generates report with:

- Pipeline execution status
- Models trained (fold completion)
- Artifacts created (file verification)
- Metrics validation (range checks)
- Errors/warnings encountered
- Pass/fail verdict

## Comparison with Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|------------------|
| Scope | Individual functions | Full pipeline |
| Data | Mock (100 rows) | Real (1,546 rows) |
| Speed | Fast (<10s) | Slower (~60s) |
| Coverage | High (80%+) | Workflow coverage |
| Catches | Logic errors | Integration bugs |
| Run Frequency | Every commit | Before merge |

**Best Practice:** Run BOTH unit and integration tests before committing.
