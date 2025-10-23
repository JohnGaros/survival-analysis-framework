# Integration Test Agent

## Purpose

This agent validates the complete end-to-end training pipeline to ensure all components work together correctly. It catches integration issues that unit tests miss.

## When to Use

Run this agent:
1. **After any code changes** to data.py, models.py, validation.py, train.py
2. **Before committing changes** to ensure nothing breaks
3. **After adding new models** to verify pipeline compatibility
4. **When changing preprocessing** steps to verify downstream effects

## What It Tests

### 1. End-to-End Pipeline Execution

**Test:** Full training pipeline completes without errors
```bash
PYTHONPATH=src python src/main.py
```

**Success Criteria:**
- ✅ All models train successfully
- ✅ No LAPACK errors
- ✅ No ValueError exceptions
- ✅ No IndexError exceptions
- ✅ Artifacts saved: model_metrics.csv, model_summary.csv
- ✅ Models saved to models/ directory

### 2. Model Training Verification

**Test:** Each model completes all CV folds
```bash
PYTHONPATH=src python -c "
from survival_framework.train import train_all_models
train_all_models('data/sample/survival_inputs_sample2000.csv')
"
```

**Success Criteria:**
- ✅ CoxPH: 5 folds completed
- ✅ Coxnet: 5 folds completed
- ✅ Weibull AFT: 5 folds completed
- ✅ GBSA: 5 folds completed
- ✅ RSF: 5 folds completed

### 3. Preprocessing Pipeline Validation

**Test:** Preprocessing handles edge cases
- Missing values in numeric features
- Missing values in categorical features
- Zero variance features
- Zero survival times
- Rare categorical levels

**Script:**
```python
import pandas as pd
import numpy as np
from survival_framework.data import make_preprocessor, make_pipeline
from survival_framework.models import CoxPHWrapper

# Create edge case data
df = pd.DataFrame({
    'debit_exp_smooth': [1.0, np.nan, 3.0, 1.0, 1.0],  # Missing + low variance
    'typeoftariff_coarse': ['A', np.nan, 'B', 'A', 'rare'],  # Missing + rare
    'survival_months': [0.0, 12.0, 6.0, 24.0, 18.0],  # Zero time
    'is_terminated': [True, False, True, False, True]
})

preprocessor = make_preprocessor(
    numeric=['debit_exp_smooth'],
    categorical=['typeoftariff_coarse']
)

# Should NOT crash
pipeline = make_pipeline(preprocessor, CoxPHWrapper())
```

### 4. Cross-Validation Fold Compatibility

**Test:** Each fold has compatible train/test splits
- Time points within test set range
- Sufficient events in each fold
- No empty predictions

### 5. Model Artifact Verification

**Test:** All expected artifacts are created
```bash
ls artifacts/model_metrics.csv
ls artifacts/model_summary.csv
ls artifacts/ph_flags.csv
ls artifacts/cox_ph/
ls artifacts/coxnet/
ls artifacts/weibull_aft/
ls artifacts/gbsa/
ls artifacts/rsf/
ls models/*.joblib
```

### 6. Performance Sanity Checks

**Test:** Model metrics are reasonable
- C-index > 0.5 (better than random)
- C-index < 1.0 (not overfitting)
- IBS > 0.0 (valid calibration)
- IBS < 0.25 (reasonable calibration)
- No NaN in metrics

## Testing Protocol

### Step 1: Clean Environment
```bash
rm -rf artifacts/ models/ data/test_outputs/
```

### Step 2: Run End-to-End
```bash
PYTHONPATH=src timeout 300 python src/main.py 2>&1 | tee integration_test.log
```

### Step 3: Check Exit Code
```bash
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully"
else
    echo "❌ Pipeline failed - check integration_test.log"
    exit 1
fi
```

### Step 4: Verify Outputs
```bash
# Check artifacts exist
test -f artifacts/model_metrics.csv || exit 1
test -f artifacts/model_summary.csv || exit 1

# Check models saved
test -d models/ || exit 1
model_count=$(ls models/*.joblib 2>/dev/null | wc -l)
if [ "$model_count" -lt 5 ]; then
    echo "❌ Expected 5+ models, found $model_count"
    exit 1
fi

echo "✅ All artifacts created"
```

### Step 5: Validate Metrics
```python
import pandas as pd

metrics = pd.read_csv('artifacts/model_metrics.csv')

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

print("✅ All metrics valid")
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

## Reporting

After running tests, generate report:

```bash
cat > INTEGRATION_TEST_REPORT.md <<EOF
# Integration Test Report

**Date:** $(date)
**Status:** [PASS/FAIL]

## Pipeline Execution

- [x] Training completed: YES/NO
- [x] Exit code: $?
- [x] Duration: XXs

## Models Trained

- [x] cox_ph: 5/5 folds
- [x] coxnet: 5/5 folds
- [x] weibull_aft: 5/5 folds
- [x] gbsa: 5/5 folds
- [x] rsf: 5/5 folds

## Artifacts Created

- [x] artifacts/model_metrics.csv
- [x] artifacts/model_summary.csv
- [x] models/*.joblib (count: X)

## Metrics Validation

- [x] C-index range: [min, max]
- [x] IBS range: [min, max]
- [x] No NaN values: YES/NO

## Errors/Warnings

[List any errors or warnings encountered]

## Conclusion

[PASS/FAIL with explanation]
EOF
```

## Usage Examples

### Example 1: After Modifying data.py

```bash
# Make changes to data.py
vim src/survival_framework/data.py

# Run integration tests
./scripts/run-integration-tests.sh

# If pass, commit
git add src/survival_framework/data.py
git commit -m "feat: improve preprocessing"
```

### Example 2: Before Creating PR

```bash
# Run full test suite
PYTHONPATH=src pytest tests/ -v

# Run integration tests
./scripts/run-integration-tests.sh

# Both pass? Create PR
gh pr create --title "Fix LAPACK error"
```

### Example 3: Continuous Integration

```yaml
# .github/workflows/test.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: PYTHONPATH=src pytest tests/ -v

      - name: Run integration tests
        run: ./scripts/run-integration-tests.sh

      - name: Upload test report
        uses: actions/upload-artifact@v3
        with:
          name: integration-test-report
          path: INTEGRATION_TEST_REPORT.md
```

## Agent Instructions

When running as an agent, execute these steps:

1. **Clean environment**: Remove old artifacts
2. **Run end-to-end**: Execute main.py with timeout
3. **Capture output**: Save stderr and stdout
4. **Check exit code**: Verify successful completion
5. **Verify artifacts**: Check all expected files exist
6. **Validate metrics**: Ensure metrics in valid ranges
7. **Generate report**: Create INTEGRATION_TEST_REPORT.md
8. **Return verdict**: PASS/FAIL with detailed findings

**Critical:** If ANY of these steps fail, report detailed error with:
- Error message
- File and line number
- Stack trace
- Suggested fix

## Comparison with Unit Tests

| Aspect | Unit Tests | Integration Tests |
|--------|-----------|------------------|
| Scope | Individual functions | Full pipeline |
| Data | Mock (100 rows) | Real (2000 rows) |
| Speed | Fast (<10s) | Slow (~60s) |
| Coverage | High (80%+) | Workflow coverage |
| Catches | Logic errors | Integration bugs |
| Run Frequency | Every commit | Before merge |

**Best Practice:** Run BOTH unit and integration tests before committing.
