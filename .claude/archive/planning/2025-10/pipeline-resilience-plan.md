# Pipeline Resilience and Logging Improvement Plan

**Date**: 2025-10-26
**Status**: Planning
**Priority**: HIGH - Critical for production reliability

## Executive Summary

After 31-hour production run failure due to MLflow crash, we need to make the survival analysis pipeline resilient to external service failures and ensure comprehensive logging for production runs.

**Current Issue**: Pipeline crashed at train.py:329 during MLflow logging, losing all aggregated metrics and preventing prediction generation despite successful model training.

---

## Problem Analysis

### 1. Current Failure Points

#### **Critical Failure (What happened)**:
```python
# train.py:329 - MLflow logging after RSF training
log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx)
# ❌ CRASH: MlflowException: Run '98bd42d60b5644b7815387dd52c24af3' not found
```

**Impact**:
- ✅ All 5 models trained (31 hours)
- ✅ Fold artifacts saved (survival curves, risk scores)
- ❌ No metrics CSV files generated
- ❌ No RSF final model saved
- ❌ No predictions generated
- ❌ No logs directory created

#### **Root Causes**:
1. **Single point of failure**: MLflow error crashes entire pipeline
2. **Late persistence**: Metrics only written to CSV at end of all training
3. **No error handling**: MLflow calls have no try/except blocks
4. **Missing logging**: Production run has no logs directory
5. **Atomic operations**: All-or-nothing approach - if one step fails, everything lost

### 2. Missing Logging Infrastructure

**Sample run** (successful):
```
data/outputs/sample/logs/
├── main_20251025_111406.log         # 18KB - Full execution log
├── performance_20251025_111406.log  # 5KB  - Clean metrics
└── warnings_20251025_111406.log     # 452B - Categorized warnings
```

**Production run** (failed):
```
data/outputs/production/logs/
└── [DOES NOT EXIST]
```

**Why this matters**:
- No way to diagnose what happened during 31-hour run
- No performance metrics captured
- No warning tracking for numerical issues
- No audit trail for production deployments

---

## Proposed Solution Architecture

### Phase 1: Incremental Metrics Persistence ✨

**Goal**: Write metrics immediately after each model completes, not at the end

#### Implementation:

```python
def _save_model_metrics_incremental(
    model_name: str,
    fold_results: List[Dict],
    artifacts_path: Path,
    logger: logging.Logger
) -> None:
    """Save metrics for a single model immediately after training.

    Creates/appends to model_metrics.csv with file locking for parallel safety.
    """
    metrics_file = artifacts_path / "model_metrics.csv"

    # Convert fold results to DataFrame
    df = pd.DataFrame(fold_results)

    # Use file locking for parallel safety
    with filelock.FileLock(f"{metrics_file}.lock"):
        if metrics_file.exists():
            # Append to existing file
            existing_df = pd.read_csv(metrics_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(metrics_file, index=False)
        else:
            # Create new file with header
            df.to_csv(metrics_file, index=False)

    logger.info(f"✅ Saved metrics for {model_name} to {metrics_file}")
    log_performance(logger, f"Metrics persisted for {model_name}",
                    avg_cindex=df['cindex'].mean(),
                    avg_ibs=df['ibs'].mean())
```

**Benefits**:
- ✅ Metrics preserved even if pipeline crashes later
- ✅ Can resume from last completed model
- ✅ Audit trail of what completed successfully
- ✅ Safe for parallel execution (file locking)

### Phase 2: MLflow Error Handling with Graceful Degradation 🛡️

**Goal**: MLflow failures should warn, not crash

#### Implementation:

```python
class SafeMLflowContext:
    """Context manager for safe MLflow operations."""

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.mlflow_available = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, mlflow.exceptions.MlflowException):
            self.logger.warning(
                f"⚠️  MLflow {self.operation} failed: {exc_val}",
                extra={"category": "mlflow_error"}
            )
            self.mlflow_available = False
            # Suppress exception - continue execution
            return True
        return False

def safe_log_metrics(metrics: Dict, step: Optional[int] = None, logger: Optional[logging.Logger] = None):
    """Log metrics to MLflow with error handling."""
    try:
        mlflow.log_metrics(metrics, step=step)
    except mlflow.exceptions.MlflowException as e:
        if logger:
            logger.warning(
                f"⚠️  MLflow metrics logging failed: {e}",
                extra={"category": "mlflow_error"}
            )
        # Continue execution - metrics already saved to CSV
    except Exception as e:
        if logger:
            logger.error(
                f"❌ Unexpected error in MLflow logging: {e}",
                extra={"category": "mlflow_error"}
            )
```

**Usage in train.py**:
```python
# Instead of direct mlflow calls:
log_metrics({f"{name}_cindex": res["cindex"]}, step=fold_idx)

# Use safe wrapper:
safe_log_metrics({f"{name}_cindex": res["cindex"]}, step=fold_idx, logger=model_logger)
```

**Benefits**:
- ✅ MLflow failures logged but don't crash pipeline
- ✅ CSV files still provide complete metrics
- ✅ Pipeline continues to prediction phase
- ✅ Warning logs alert to MLflow issues

### Phase 3: Comprehensive Production Logging 📊

**Goal**: Production runs must have same logging as sample runs

#### Implementation:

**A. Ensure log directory creation**:
```python
# In main.py, BEFORE train_all_models():
def setup_logging_infrastructure(run_type: str) -> Path:
    """Create logs directory and setup logging before pipeline starts."""
    paths = get_output_paths(run_type)
    log_dir = Path(paths["logs"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging config (already implemented)
    logger = setup_logging(log_dir, run_type)
    logger.info(f"📁 Logging infrastructure initialized: {log_dir}")

    return log_dir
```

**B. Log directory verification**:
```python
# Add to train_all_models() start:
def train_all_models(...):
    """Train all models with comprehensive logging."""
    logger = logging.getLogger("survival_framework.train")

    # VERIFY logs directory exists
    paths = get_output_paths(run_type)
    log_dir = Path(paths["logs"])
    if not log_dir.exists():
        logger.warning(f"⚠️  Logs directory missing: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created logs directory: {log_dir}")

    logger.info(f"Starting training pipeline for run_type={run_type}")
    logger.info(f"Logging to: {log_dir}")
```

**C. Performance tracking per model**:
```python
# After each model completes:
log_performance(
    logger,
    f"{model_name} training complete",
    duration_minutes=round(elapsed_time / 60, 2),
    avg_cindex=round(np.mean([r['cindex'] for r in model_results]), 4),
    avg_ibs=round(np.mean([r['ibs'] for r in model_results]), 4),
    n_folds=len(model_results)
)
```

**Benefits**:
- ✅ Production runs have full audit trail
- ✅ Performance metrics tracked per model
- ✅ Can diagnose failures post-mortem
- ✅ Warnings categorized for analysis

### Phase 4: Recovery Mechanism for Interrupted Pipelines 🔄

**Goal**: Resume from last completed model without retraining

#### Implementation:

```python
def detect_completed_models(artifacts_path: Path, models: Dict) -> Tuple[List[str], List[str]]:
    """Detect which models have already been trained.

    Returns:
        Tuple of (completed_models, pending_models)
    """
    completed = []
    pending = []

    for model_name in models.keys():
        # Check if model has all 5 fold artifacts
        model_dir = artifacts_path / model_name
        if not model_dir.exists():
            pending.append(model_name)
            continue

        # Check for fold artifacts (risk and survival files)
        fold_files = list(model_dir.glob(f"{model_name}_fold*_risk.npy"))
        if len(fold_files) == 5:  # All folds complete
            completed.append(model_name)
        else:
            pending.append(model_name)

    return completed, pending

def train_all_models_with_resume(..., resume: bool = True):
    """Train all models with resume capability."""
    logger = logging.getLogger("survival_framework.train")

    if resume:
        completed, pending = detect_completed_models(paths["artifacts"], models)

        if completed:
            logger.info(f"✅ Found {len(completed)} completed models: {completed}")
            logger.info(f"⏭️  Skipping completed models")

            # Load existing metrics
            metrics_file = Path(paths["artifacts"]) / "model_metrics.csv"
            if metrics_file.exists():
                existing_metrics = pd.read_csv(metrics_file)
                logger.info(f"📊 Loaded {len(existing_metrics)} existing metric rows")
        else:
            logger.info("🆕 No completed models found, starting fresh")
            pending = list(models.keys())
    else:
        pending = list(models.keys())

    # Only train pending models
    for model_name in pending:
        logger.info(f"🏋️ Training {model_name}...")
        # ... existing training code ...
```

**Benefits**:
- ✅ Can resume 31-hour run from failure point
- ✅ Don't waste completed training
- ✅ Faster iteration when debugging
- ✅ Explicit resume flag for control

### Phase 5: Model Summary Regeneration 📈

**Goal**: Regenerate summary from existing metrics CSV

#### Implementation:

```python
def regenerate_model_summary(artifacts_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Regenerate model_summary.csv from model_metrics.csv."""
    metrics_file = artifacts_path / "model_metrics.csv"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Cannot regenerate summary: {metrics_file} not found")

    # Load metrics
    df = pd.read_csv(metrics_file)
    logger.info(f"📊 Loaded {len(df)} rows from {metrics_file}")

    # Aggregate by model
    summary = df.groupby("model").agg({
        "cindex": ["mean", "std"],
        "ibs": ["mean", "std"],
        "fold": "count"
    }).reset_index()

    # Flatten column names
    summary.columns = ["model", "cindex_mean", "cindex_std", "ibs_mean", "ibs_std", "n_folds"]

    # Rank models by C-index (higher is better)
    summary["rank"] = summary["cindex_mean"].rank(ascending=False, method="min")
    summary = summary.sort_values("rank")

    # Save
    summary_file = artifacts_path / "model_summary.csv"
    summary.to_csv(summary_file, index=False)
    logger.info(f"✅ Model summary saved to {summary_file}")

    return summary
```

**Benefits**:
- ✅ Can recover rankings from existing metrics
- ✅ Supports recovery script
- ✅ Verifies metric integrity

---

## Implementation Plan

### Task 1: Add Incremental Metrics Persistence
**Priority**: HIGH
**Estimated time**: 2 hours

**Changes**:
1. Add `filelock` to requirements.txt
2. Create `_save_model_metrics_incremental()` in train.py
3. Call after each model completes training
4. Update CHANGELOG.md

**Files modified**:
- `src/survival_framework/train.py`
- `requirements.txt`

### Task 2: Implement MLflow Error Handling
**Priority**: HIGH
**Estimated time**: 1 hour

**Changes**:
1. Create `SafeMLflowContext` class in tracking.py
2. Create `safe_log_metrics()`, `safe_log_params()`, `safe_log_artifact()` wrappers
3. Replace all direct mlflow calls with safe wrappers
4. Update CHANGELOG.md

**Files modified**:
- `src/survival_framework/tracking.py`
- `src/survival_framework/train.py`

### Task 3: Fix Production Logging
**Priority**: HIGH
**Estimated time**: 30 minutes

**Changes**:
1. Add `setup_logging_infrastructure()` to main.py
2. Call before train_all_models()
3. Verify logs directory in train_all_models()
4. Update CHANGELOG.md

**Files modified**:
- `src/main.py`
- `src/survival_framework/train.py`

### Task 4: Add Resume Capability
**Priority**: MEDIUM
**Estimated time**: 2 hours

**Changes**:
1. Create `detect_completed_models()` in train.py
2. Add `--resume` CLI flag to main.py
3. Skip completed models if resume=True
4. Update CHANGELOG.md

**Files modified**:
- `src/survival_framework/train.py`
- `src/main.py`

### Task 5: Create Recovery Script
**Priority**: MEDIUM
**Estimated time**: 1 hour

**Changes**:
1. Create `scripts/recover_production_metrics.py`
2. Regenerate model_summary.csv from model_metrics.csv
3. Train final RSF model on full dataset
4. Generate predictions using best model
5. Update CHANGELOG.md

**Files created**:
- `scripts/recover_production_metrics.py`

### Task 6: Testing
**Priority**: HIGH
**Estimated time**: 2 hours

**Changes**:
1. Test incremental metrics with sample data
2. Test MLflow failure simulation
3. Test resume functionality
4. Test recovery script
5. Verify production logging works

---

## Success Criteria

### Must Have (Before Production)
- ✅ Metrics CSV files written incrementally
- ✅ MLflow failures don't crash pipeline
- ✅ Production runs create logs directory
- ✅ Can resume from interruption
- ✅ Recovery script reconstructs metrics

### Nice to Have (Future)
- ⏭️  Checkpoint final models incrementally
- ⏭️  Auto-detect MLflow unavailability at startup
- ⏭️  Email/Slack notifications on failure
- ⏭️  Distributed training with fault tolerance

---

## Risk Assessment

### High Risk (Addressed)
1. **31-hour training lost** → Incremental persistence + resume
2. **No production logs** → Forced directory creation
3. **MLflow single point of failure** → Error handling wrappers

### Medium Risk (Mitigated)
1. **File locking overhead** → Minimal (only during CSV writes)
2. **Resume logic bugs** → Comprehensive testing required
3. **Partial artifacts** → Verify all 5 folds before marking complete

### Low Risk (Acceptable)
1. **Slightly slower execution** → Negligible (I/O during training)
2. **Code complexity** → Well-documented, tested

---

## Timeline

**Total estimated time**: 8.5 hours

**Week 1** (Tasks 1-3):
- Day 1: Incremental metrics persistence (2h)
- Day 2: MLflow error handling (1h)
- Day 3: Production logging fix (0.5h)

**Week 2** (Tasks 4-6):
- Day 1: Resume capability (2h)
- Day 2: Recovery script (1h)
- Day 3: Testing (2h)

---

## References

- Current failure: `train.py:329` - MLflow log_metrics crash
- Logging policy: `.claude/skills/logging-policy.md`
- Production artifacts: `data/outputs/production/artifacts/`
- Sample logs: `data/outputs/sample/logs/`

---

**Next Steps**: Present plan to user for approval, then implement Task 1-3 (high priority).
