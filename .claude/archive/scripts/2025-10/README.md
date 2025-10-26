# Archive: October 2025 - One-Off Scripts

## Contents

### Manual Recovery Scripts

These scripts were created for one-time manual recovery tasks. The functionality has been replaced by the automatic recovery system integrated into the main pipeline.

---

#### `recover_pipeline.py`
- **Archived**: 2025-10-26
- **Size**: 10.2 KB
- **Purpose**: Manual recovery script for fitting final models after pipeline crash
- **Functionality**:
  - Load full dataset
  - Fit preprocessing pipeline
  - Train final GBSA model on complete data
  - Save model with versioned filename
- **Usage Context**: Used once after 31-hour production run crashed
- **Status**: ✅ **Replaced by automatic recovery in src/survival_framework/recovery.py**
- **Why Archived**: Automatic recovery system now handles this automatically

---

#### `test_recovery.py`
- **Archived**: 2025-10-26
- **Size**: 2.1 KB
- **Purpose**: Test script for validating recovery system detection logic
- **Functionality**:
  - Test `_detect_completed_models()` with partial metrics
  - Test `_generate_metrics_summary_from_partial()` with 3 models
  - Create test scenario with incomplete model_metrics.csv
- **Status**: ✅ **Testing Complete, Recovery System Validated**
- **Why Archived**: One-time validation completed successfully

---

## Replacement

These manual scripts are no longer needed because:

1. **Automatic Recovery System** (`src/survival_framework/recovery.py`)
   - Automatically detects completed models from incremental metrics
   - Fits final models for completed models only
   - Generates predictions using best available model
   - Integrated into main.py exception handler
   - Triggered automatically on any pipeline failure

2. **Integration**:
   ```python
   # In src/main.py
   except Exception as e:
       logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

       # Attempt automatic recovery
       recovery_successful = attempt_recovery(
           run_type=run_type,
           input_file=input_file,
           logger=logger,
           original_error=e
       )

       if recovery_successful:
           return 0  # Partial success
       else:
           return 1  # Total failure
   ```

## Historical Context

**Problem Solved:**
After a 31-hour production training run crashed before saving final models, these scripts were created to manually recover the work. The experience led to designing and implementing the comprehensive automatic recovery system that prevents this scenario in the future.

**Commits:**
- bf6ef80 - feat: Add comprehensive pipeline resilience and automatic recovery system

## Retrieval

To view archived scripts:
```bash
cat .claude/archive/scripts/2025-10/recover_pipeline.py
cat .claude/archive/scripts/2025-10/test_recovery.py
```

These scripts are preserved for:
- Historical reference
- Understanding the problem that led to automatic recovery
- Examples of manual recovery approaches
- Comparison with automatic solution

## References

- **src/survival_framework/recovery.py** - Automatic recovery implementation
- **CHANGELOG.md** - Recovery system documentation
- **.claude/assessments/auto-recovery-design.md** - Archived design document
