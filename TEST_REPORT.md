# Test Report - Initial Run

**Date:** 2025-10-23
**Command:** `PYTHONPATH=src pytest tests/ -v`
**Result:** 27 passed, 10 failed, 42.51% coverage

## Summary

- ✅ **Passed:** 27/37 tests (73%)
- ❌ **Failed:** 10/37 tests (27%)
- 📊 **Coverage:** 42.51% (target: >80%)
- ⏱️ **Duration:** 3.59s

## Test Results by Module

### ✅ test_data.py: 12/13 passed (92%)

**Passed Tests:**
- TestToStructuredY: All 3 tests passed
  - ✓ Basic conversion
  - ✓ Boolean event conversion
  - ✓ Time float conversion
- TestMakePreprocessor: All 3 tests passed
  - ✓ Default columns
  - ✓ Custom columns
  - ✓ Fit transform
- TestSplitXY: All 5 tests passed
  - ✓ Basic split
  - ✓ Feature columns
  - ✓ Structured array format
  - ✓ Dropna true
  - ✓ Dropna false
- TestMakePipeline: 1/2 tests passed
  - ✓ Pipeline creation

**Failed Tests:**
1. ❌ `test_pipeline_fit` - ValueError: LAPACK error
   - **Root Cause:** Small sample size (100 rows) causing numerical instability in CoxPH
   - **Fix:** Use larger sample or mock the model fit

### ❌ test_metrics.py: 1/12 passed (8%)

**Failed Tests:**
All tests failing due to API mismatches:

1. **C-index tests (3 failures):**
   - ❌ `test_cindex_range`
   - ❌ `test_perfect_discrimination`
   - ❌ `test_cindex_with_censoring`
   - **Root Cause:** `concordance_index_ipcw` returns 5 values, not 2
   - **Fix:** Update to unpack all 5 values: `cindex, concordant, discordant, tied_risk, tied_time = ...`

2. **IBS tests (3 failures):**
   - ❌ `test_ibs_range`
   - ❌ `test_ibs_shape_validation`
   - ❌ `test_ibs_perfect_predictions`
   - **Root Cause:** Time points exceed follow-up time range
   - **Fix:** Adjust test time points to be within data range

3. **Time-dependent AUC tests (2 failures):**
   - ❌ `test_auc_output_shape`
   - ❌ `test_auc_range`
   - **Root Cause:** Same as IBS - time points out of range
   - **Fix:** Adjust test time points

**Passed Tests:**
- ✓ `test_auc_with_good_discrimination`

### ✅ test_utils.py: 14/15 passed (93%)

**Passed Tests:**
- TestEnsureDir: All 3 tests passed
- TestDefaultTimeGrid: All 5 tests passed
- TestSaveModelMetrics: All 3 tests passed
- TestVersionedName: 3/4 tests passed

**Failed Tests:**
1. ❌ `test_timestamp_format`
   - **Root Cause:** Timestamp format is `YYYYMMDD_HHMMSS` which splits into 4 parts, not 3
   - **Expected:** `['cox', 'ph', 'YYYYMMDDHHMMSS']` (3 parts)
   - **Actual:** `['cox', 'ph', '20251023', '150246']` (4 parts due to underscore in timestamp)
   - **Fix:** Update test to expect 4 parts or remove underscore from timestamp

## Coverage Analysis

### Modules by Coverage

| Module | Coverage | Missing Lines |
|--------|----------|---------------|
| `data.py` | 100% | - |
| `utils.py` | 100% | - |
| `__init__.py` | 100% | - |
| `metrics.py` | 84.62% | 35, 64 |
| `models.py` | 56.12% | Many model methods |
| `tracking.py` | 0% | All lines |
| `train.py` | 0% | All lines |
| `validation.py` | 0% | All lines |

### Missing Coverage

**Critical gaps:**
1. **models.py (56% coverage)** - Need tests for:
   - All model wrappers (CoxnetWrapper, StratifiedCoxWrapper, etc.)
   - Model fitting and prediction methods
   - Edge cases and error handling

2. **validation.py (0% coverage)** - Need tests for:
   - `CVConfig` dataclass
   - `event_balanced_splitter()`
   - `ph_assumption_flags()`
   - `evaluate_model()`

3. **tracking.py (0% coverage)** - Need tests for:
   - MLflow integration functions
   - Artifact logging
   - Parameter tracking

4. **train.py (0% coverage)** - Need integration tests for:
   - Full training pipeline
   - Model building
   - Cross-validation workflow

## Issues Found

### 1. Import Path Configuration

**Issue:** Tests couldn't find `survival_framework` module
**Solution:** Need to set `PYTHONPATH=src` or install package in editable mode
**Action:** Add to documentation and pytest configuration

### 2. API Mismatch in metrics.py

**Issue:** `concordance_index_ipcw` returns 5 values, not 2
**Current Code:**
```python
cindex, _ = concordance_index_ipcw(y_train, y_test, risk_scores)
```

**Should be:**
```python
cindex, concordant, discordant, tied_risk, tied_time = concordance_index_ipcw(...)
```

### 3. Test Data Time Ranges

**Issue:** Test time points exceed data range
**Solution:** Ensure time points are within `[min_time, max_time)` of test data

### 4. Small Sample Size Issues

**Issue:** 100 rows insufficient for CoxPH numerical stability
**Solution:** Use larger samples or mock model fitting in unit tests

### 5. Timestamp Format

**Issue:** `versioned_name()` format doesn't match test expectations
**Current:** `base_YYYYMMDD_HHMMSS` (splits into 4 parts)
**Test expects:** 3 parts
**Solution:** Update test to match actual format

## Recommendations

### Immediate Fixes (High Priority)

1. **Fix metrics.py API usage:**
   ```python
   # In src/survival_framework/metrics.py
   def compute_cindex(y_train, y_test, risk_scores) -> float:
       result = concordance_index_ipcw(y_train, y_test, risk_scores)
       cindex = result[0]  # First element is the C-index
       return float(cindex)
   ```

2. **Fix test time ranges:**
   - Update test fixtures to use time points within data range
   - Or adjust compute_ibs/compute_time_dependent_auc calls

3. **Fix timestamp test:**
   - Update assertion to expect 4 parts instead of 3

4. **Configure PYTHONPATH:**
   - Add to pytest.ini or setup.py
   - Document in TESTING.md

### Medium Priority

5. **Add model tests:**
   - Create test_models.py with tests for each wrapper
   - Test fit, predict, score methods
   - Use mocking for expensive operations

6. **Add validation tests:**
   - Create integration tests for CV workflow
   - Test event-balanced splitting
   - Test PH assumption checking

7. **Add tracking tests:**
   - Mock MLflow to test logging functions
   - Verify parameter and metric logging

### Long Term

8. **Integration tests:**
   - End-to-end training pipeline test
   - Full workflow with sample data
   - Performance benchmarks

9. **Increase coverage to >80%:**
   - Focus on models.py (currently 56%)
   - Add validation.py tests (currently 0%)
   - Add tracking.py tests (currently 0%)

10. **CI/CD setup:**
    - GitHub Actions workflow
    - Automated test runs on PR
    - Coverage reports

## Next Steps

1. Run: `PYTHONPATH=src pytest tests/test_metrics.py::TestComputeCindex -v`
2. Fix the API mismatch in compute_cindex
3. Fix test time range issues
4. Re-run full suite
5. Add missing test files (test_models.py, test_validation.py)
6. Reach 80% coverage target

## Commands to Run

```bash
# Fix and re-run specific tests
PYTHONPATH=src pytest tests/test_metrics.py -v

# Check coverage for specific module
PYTHONPATH=src pytest tests/ --cov=src/survival_framework/metrics --cov-report=term-missing

# Run only passing tests
PYTHONPATH=src pytest tests/test_utils.py tests/test_data.py -v

# Generate full coverage report
PYTHONPATH=src pytest tests/ --cov=src/survival_framework --cov-report=html:data/test_outputs/coverage
```

## Test Infrastructure Status

✅ **Working Well:**
- Fixture setup (conftest.py)
- Test organization by module
- Coverage reporting
- Test categorization

⚠️ **Needs Improvement:**
- PYTHONPATH configuration
- Test data ranges
- API compatibility
- Coverage gaps in core modules
