# Pipeline Resilience Implementation Status

**Date**: 2025-10-26
**Session**: High-priority resilience tasks

## ✅ Completed Tasks

### 1. Added filelock Dependency
**File**: `requirements.txt`
**Changes**:
```python
filelock>=3.12  # For parallel-safe CSV writes
```

### 2. Implemented Safe MLflow Wrappers
**File**: `src/survival_framework/tracking.py`
**Changes Added**:
- `safe_log_metrics()` - Logs metrics with error handling
- `safe_log_params()` - Logs parameters with error handling
- `safe_log_artifact()` - Logs artifacts with error handling

**Key Features**:
- Returns bool (success/failure) instead of raising exceptions
- Logs warnings to logger with category="mlflow_error"
- Continues execution even if MLflow fails
- Preserves CSV-based metrics as primary source of truth

---

## 🔨 Remaining High-Priority Tasks

### Task 3: Implement Incremental Metrics Persistence

**File**: `src/survival_framework/train.py`

**Required Changes**:

#### A. Add imports at top of file:
```python
import filelock
from pathlib import Path
```

#### B. Add function before `train_all_models()`:
```python
def _save_model_metrics_incremental(
    model_name: str,
    fold_results: List[Dict],
    artifacts_path: str,
    logger: logging.Logger
) -> None:
    """Save metrics for a single model immediately after training.

    Creates/appends to model_metrics.csv with file locking for parallel safety.

    Args:
        model_name: Name of the model (e.g., "cox_ph", "gbsa")
        fold_results: List of metric dictionaries from cross-validation folds
        artifacts_path: Path to artifacts directory
        logger: Logger instance for status messages

    Example:
        >>> fold_results = [
        ...     {"model": "cox_ph", "fold": 0, "cindex": 0.742, "ibs": 0.152},
        ...     {"model": "cox_ph", "fold": 1, "cindex": 0.738, "ibs": 0.148}
        ... ]
        >>> _save_model_metrics_incremental("cox_ph", fold_results, "artifacts", logger)
    """
    metrics_file = Path(artifacts_path) / "model_metrics.csv"
    lock_file = Path(artifacts_path) / ".model_metrics.csv.lock"

    # Convert fold results to DataFrame
    df = pd.DataFrame(fold_results)

    # Use file locking for parallel safety
    lock = filelock.FileLock(lock_file, timeout=30)
    try:
        with lock:
            if metrics_file.exists():
                # Append to existing file
                existing_df = pd.read_csv(metrics_file)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(metrics_file, index=False)
                logger.info(f"✅ Appended {len(df)} rows to {metrics_file}")
            else:
                # Create new file with header
                df.to_csv(metrics_file, index=False)
                logger.info(f"✅ Created {metrics_file} with {len(df)} rows")

        # Log performance summary
        avg_cindex = df['cindex'].mean()
        avg_ibs = df['ibs'].mean()
        logger.info(
            f"📊 {model_name} metrics: C-index={avg_cindex:.4f}, IBS={avg_ibs:.4f}"
        )

    except filelock.Timeout:
        logger.error(f"❌ File lock timeout for {metrics_file}")
        raise
```

#### C. Modify `train_all_models()` to call incremental save:

**Find this section** (around line 354-360):
```python
# Compute average metrics for this model
model_results = [r for r in rows if r["model"] == name]
avg_cindex = np.mean([r["cindex"] for r in model_results])
avg_ibs = np.mean([r["ibs"] for r in model_results])
model_logger.info(
    f"{name} average: cindex={avg_cindex:.4f}, ibs={avg_ibs:.4f}"
)
```

**Add immediately after**:
```python
# Save metrics incrementally to CSV
_save_model_metrics_incremental(
    model_name=name,
    fold_results=model_results,
    artifacts_path=paths["artifacts"],
    logger=model_logger
)
```

---

### Task 4: Update train.py to Use Safe MLflow Wrappers

**File**: `src/survival_framework/train.py`

**Required Changes**:

#### A. Update imports:
```python
from .tracking import (
    start_run, log_params, log_metrics, log_artifact,
    safe_log_metrics, safe_log_params, safe_log_artifact  # Add safe versions
)
```

#### B. Replace MLflow calls with safe versions:

**Location 1** - Around line 279-285 (log run params):
```python
# OLD:
log_params({
    "run_type": run_type,
    "input_file": file_path,
    "n_samples": len(df),
    "execution_mode": execution_config.mode.value,
    "n_jobs": execution_config.n_jobs
})

# NEW:
safe_log_params({
    "run_type": run_type,
    "input_file": file_path,
    "n_samples": len(df),
    "execution_mode": execution_config.mode.value,
    "n_jobs": execution_config.n_jobs
}, logger=logger)
```

**Location 2** - Around line 286 (log PH flags artifact):
```python
# OLD:
log_artifact(ph_flags_path)

# NEW:
safe_log_artifact(ph_flags_path, logger=logger)
```

**Location 3** - Around line 329 and 350 (log fold metrics):
```python
# OLD (parallel execution):
log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx)

# NEW:
safe_log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx, logger=model_logger)

# OLD (sequential execution):
log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx)

# NEW:
safe_log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx, logger=model_logger)
```

**Location 4** - Around line 383 (log final artifacts):
```python
# OLD:
log_artifact(metrics_path)
log_artifact(summary_path)

# NEW:
safe_log_artifact(metrics_path, logger=logger)
safe_log_artifact(summary_path, logger=logger)
```

---

### Task 5: Fix Production Logging Infrastructure

**File**: `src/main.py`

**Required Changes**:

#### A. Add logging setup before training:

**Find** (around line 120-125):
```python
def run_pipeline(...):
    # Train models
    logger.info("Starting model training")
    with Timer(logger, "Model training"):
        train_all_models(...)
```

**Add BEFORE `train_all_models()` call**:
```python
# Ensure logs directory exists for production runs
paths = get_output_paths(run_type)
log_dir = Path(paths["logs"])
if not log_dir.exists():
    logger.warning(f"⚠️  Logs directory missing: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Created logs directory: {log_dir}")
else:
    logger.info(f"📁 Logs directory verified: {log_dir}")
```

#### B. Add imports at top:
```python
from pathlib import Path
from survival_framework.utils import get_output_paths
```

---

## Testing Plan

### 1. Unit Test for Incremental Metrics
**File**: `tests/test_train.py` (create if doesn't exist)

```python
import tempfile
from pathlib import Path
import pandas as pd
from survival_framework.train import _save_model_metrics_incremental
import logging

def test_incremental_metrics_persistence():
    """Test that metrics are saved incrementally with file locking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = logging.getLogger("test")

        # First model
        fold_results_1 = [
            {"model": "cox_ph", "fold": 0, "cindex": 0.742, "ibs": 0.152},
            {"model": "cox_ph", "fold": 1, "cindex": 0.738, "ibs": 0.148}
        ]
        _save_model_metrics_incremental("cox_ph", fold_results_1, tmpdir, logger)

        # Verify file created
        metrics_file = Path(tmpdir) / "model_metrics.csv"
        assert metrics_file.exists()
        df1 = pd.read_csv(metrics_file)
        assert len(df1) == 2
        assert list(df1["model"]) == ["cox_ph", "cox_ph"]

        # Second model (append)
        fold_results_2 = [
            {"model": "gbsa", "fold": 0, "cindex": 0.752, "ibs": 0.142}
        ]
        _save_model_metrics_incremental("gbsa", fold_results_2, tmpdir, logger)

        # Verify appended
        df2 = pd.read_csv(metrics_file)
        assert len(df2) == 3
        assert list(df2["model"]) == ["cox_ph", "cox_ph", "gbsa"]
```

### 2. Integration Test with Sample Data
```bash
# Run with sample data to verify:
# 1. Metrics saved incrementally
# 2. MLflow failures don't crash pipeline
# 3. Logs directory created

python src/main.py --run-type sample --execution-mode sequential
```

**Expected results**:
- ✅ `data/outputs/sample/artifacts/model_metrics.csv` grows after each model
- ✅ `data/outputs/sample/logs/` directory contains log files
- ✅ Pipeline completes successfully even if MLflow fails

### 3. Simulate MLflow Failure
```python
# In train.py, temporarily modify safe_log_metrics to always fail:
def safe_log_metrics(...):
    if logger:
        logger.warning("SIMULATED MLflow failure")
    return False
```

**Expected results**:
- ✅ Warning logs show "MLflow metrics logging failed"
- ✅ Pipeline continues to completion
- ✅ CSV files contain complete metrics
- ✅ Predictions generated successfully

---

## Success Criteria

Before considering this task complete:

- [ ] `requirements.txt` includes filelock dependency
- [ ] `tracking.py` has safe_log_* wrapper functions
- [ ] `train.py` saves metrics incrementally after each model
- [ ] `train.py` uses safe_log_* wrappers instead of direct mlflow calls
- [ ] `main.py` ensures logs directory exists
- [ ] Unit test passes for incremental metrics
- [ ] Integration test passes with sample data
- [ ] MLflow failure simulation test passes
- [ ] CHANGELOG.md updated with changes

---

## Next Steps for Continuation

1. **Complete Task 3**: Add incremental metrics persistence to train.py
2. **Complete Task 4**: Replace all MLflow calls with safe wrappers
3. **Complete Task 5**: Fix production logging in main.py
4. **Run tests**: Verify with sample data
5. **Update CHANGELOG.md**: Document all changes
6. **Commit changes**: Create commit with detailed message
7. **Optional**: Create recovery script for current production run

---

## Notes

- ✅ Tasks 1-2 complete (filelock, safe wrappers)
- ⏳ Tasks 3-5 need implementation (detailed instructions above)
- 📊 Full plan available in `pipeline-resilience-plan.md`
- 🎯 Total estimated remaining time: ~2 hours
