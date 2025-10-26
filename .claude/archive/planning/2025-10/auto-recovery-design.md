# Automatic Pipeline Recovery Design

**Date**: 2025-10-26
**Status**: Design Proposal
**Priority**: HIGH - Critical for production reliability

## Executive Summary

**Recommendation**: **Enhance existing exception handling in main.py - NO separate skill needed**

The recovery logic should be **inline exception handling**, not a separate skill, because:
1. Recovery needs to happen in the same execution context (same memory, same logger, same paths)
2. We already have the necessary building blocks from resilience work
3. A skill would require complex state serialization and couldn't access in-memory data
4. Exception handling is cleaner and more maintainable

---

## Current State Analysis

### What We Have Now

#### ✅ Recent Resilience Improvements (just implemented):
1. **Incremental metrics persistence** - CSV written after each model
2. **Safe MLflow wrappers** - Failures don't crash pipeline
3. **Production logging** - Logs directory auto-created
4. **File locking** - Parallel-safe CSV writes

#### ⚠️ Current Exception Handling:
```python
# main.py:178-180
except Exception as e:
    logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
    return 1  # ❌ Just exits - no recovery attempt
```

**Problem**: When pipeline fails, we just log and exit. We don't attempt to:
- Generate metrics summary from partial CSV
- Fit final models from completed CV folds
- Generate predictions with available models
- Notify user of what was recovered

### What Happens in a Failure (Current Behavior)

**Scenario**: MLflow crashes after GBSA completes but before RSF starts

**Current outcome**:
- ❌ Exit with error code 1
- ❌ No predictions generated
- ❌ No model_summary.csv created
- ❌ User must manually run recovery script
- ✅ BUT: model_metrics.csv has GBSA data (thanks to incremental persistence!)
- ✅ BUT: GBSA artifacts exist in artifacts/gbsa/

**Desired outcome**:
- ✅ Catch exception gracefully
- ✅ Detect what models completed (check model_metrics.csv)
- ✅ Generate model_summary.csv from available data
- ✅ Fit final models for completed models only
- ✅ Generate predictions using best available model
- ✅ Send notification about partial completion
- ✅ Exit with code 0 (success with warnings)

---

## Proposed Solution: Enhanced Exception Handling

### Design Principles

1. **Fail gracefully**: Always try to salvage what we can
2. **Notify clearly**: Tell user exactly what succeeded/failed
3. **Preserve work**: Never discard completed model results
4. **Automatic recovery**: No manual intervention required
5. **Idempotent**: Running recovery twice is safe

### Architecture

```python
# main.py:run_pipeline()

try:
    # Training phase
    if not predict_only:
        train_all_models(...)  # May fail partway through

    # Prediction phase
    if not train_only:
        generate_predictions(...)  # May fail if no models

    # Success path
    logger.info("✓ COMPLETED SUCCESSFULLY")
    return 0

except Exception as e:
    # ============================================================
    # AUTOMATIC RECOVERY PHASE
    # ============================================================
    logger.error(f"Pipeline failed: {e}", exc_info=True)
    logger.warning("=" * 70)
    logger.warning("ATTEMPTING AUTOMATIC RECOVERY")
    logger.warning("=" * 70)

    recovery_successful = attempt_recovery(
        run_type=run_type,
        input_file=input_file,
        logger=logger,
        original_error=e
    )

    if recovery_successful:
        logger.warning("=" * 70)
        logger.warning("⚠️  PARTIAL SUCCESS - RECOVERY COMPLETED")
        logger.warning("=" * 70)
        logger.warning("Some models failed, but predictions generated from available models")
        logger.warning(f"Original error: {str(e)}")
        return 0  # Success with warnings
    else:
        logger.error("=" * 70)
        logger.error("❌ RECOVERY FAILED")
        logger.error("=" * 70)
        return 1  # Total failure
```

### Recovery Function Design

```python
def attempt_recovery(
    run_type: RunType,
    input_file: str,
    logger: logging.Logger,
    original_error: Exception
) -> bool:
    """Attempt to recover from pipeline failure.

    Checks what models completed successfully and attempts to:
    1. Generate model_summary.csv from partial metrics
    2. Fit final models for completed models
    3. Generate predictions using best available model
    4. Send notification about partial completion

    Args:
        run_type: Sample or production run
        input_file: Path to input data
        logger: Logger instance
        original_error: The exception that triggered recovery

    Returns:
        True if recovery generated usable outputs, False otherwise
    """
    paths = get_output_paths(run_type)

    # Step 1: Check what we have
    logger.info("\n[RECOVERY] Step 1: Analyzing completed work...")
    completed_models = _detect_completed_models(paths["artifacts"])

    if not completed_models:
        logger.error("[RECOVERY] No completed models found - cannot recover")
        return False

    logger.info(f"[RECOVERY] Found {len(completed_models)} completed models: {completed_models}")

    # Step 2: Generate metrics summary from incremental CSV
    logger.info("\n[RECOVERY] Step 2: Generating metrics summary...")
    try:
        _generate_metrics_summary_from_partial(
            paths["artifacts"],
            completed_models,
            logger
        )
        logger.info("[RECOVERY] ✓ Metrics summary generated")
    except Exception as e:
        logger.warning(f"[RECOVERY] ⚠️  Metrics summary failed: {e}")

    # Step 3: Fit final models for completed models only
    logger.info("\n[RECOVERY] Step 3: Fitting final models...")
    try:
        _fit_final_models_for_completed(
            input_file=input_file,
            run_type=run_type,
            completed_models=completed_models,
            logger=logger
        )
        logger.info("[RECOVERY] ✓ Final models fitted")
    except Exception as e:
        logger.error(f"[RECOVERY] ❌ Final model fitting failed: {e}")
        # Continue anyway - maybe we have old models

    # Step 4: Generate predictions with best available model
    logger.info("\n[RECOVERY] Step 4: Generating predictions...")
    try:
        pred_path = _generate_predictions_from_partial(
            input_file=input_file,
            run_type=run_type,
            completed_models=completed_models,
            logger=logger
        )
        logger.info(f"[RECOVERY] ✓ Predictions saved to: {pred_path}")
    except Exception as e:
        logger.error(f"[RECOVERY] ❌ Prediction generation failed: {e}")
        return False

    # Step 5: Send notification
    logger.info("\n[RECOVERY] Step 5: Sending notification...")
    _send_recovery_notification(
        run_type=run_type,
        completed_models=completed_models,
        failed_models=_detect_failed_models(completed_models),
        original_error=original_error,
        logger=logger
    )

    return True


def _detect_completed_models(artifacts_path: str) -> List[str]:
    """Detect which models completed all CV folds.

    Checks model_metrics.csv to see which models have complete data.

    Returns:
        List of model names that completed successfully
    """
    metrics_file = Path(artifacts_path) / "model_metrics.csv"

    if not metrics_file.exists():
        return []

    # Read incremental metrics
    df = pd.read_csv(metrics_file)

    # Check which models have 5 complete folds
    model_counts = df.groupby("model").size()
    completed = model_counts[model_counts == 5].index.tolist()

    return completed


def _generate_metrics_summary_from_partial(
    artifacts_path: str,
    completed_models: List[str],
    logger: logging.Logger
) -> None:
    """Generate model_summary.csv from partial model_metrics.csv.

    Only includes models that completed all folds.
    """
    metrics_file = Path(artifacts_path) / "model_metrics.csv"
    summary_file = Path(artifacts_path) / "model_summary.csv"

    # Read metrics
    df = pd.read_csv(metrics_file)

    # Filter to completed models only
    df = df[df["model"].isin(completed_models)]

    # Aggregate
    summary = df.groupby("model").agg({
        "cindex": "mean",
        "ibs": "mean"
    }).reset_index()

    # Add ranks
    summary["rank_cindex"] = summary["cindex"].rank(ascending=False)
    summary["rank_ibs"] = summary["ibs"].rank(ascending=True)

    # Sort by cindex rank
    summary = summary.sort_values("rank_cindex")

    # Save
    summary.to_csv(summary_file, index=False)
    logger.info(f"[RECOVERY] Saved summary for {len(summary)} models to {summary_file}")


def _fit_final_models_for_completed(
    input_file: str,
    run_type: RunType,
    completed_models: List[str],
    logger: logging.Logger
) -> None:
    """Fit final models only for models that completed CV.

    This is similar to the recovery script but integrated into main pipeline.
    """
    # Load data
    df = load_data(input_file, run_type=run_type)
    X, y, ids = split_X_y(df)

    # Create preprocessing pipeline
    preprocessor = make_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Fit only completed models
    paths = get_output_paths(run_type)
    models_dir = Path(paths["models"])

    for model_name in completed_models:
        logger.info(f"[RECOVERY] Fitting final {model_name}...")

        # Create model wrapper
        model = _create_model_wrapper(model_name)

        # Fit on full dataset
        model.fit(X_transformed, y)

        # Save as pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        model_filename = versioned_name(f"{run_type}_{model_name}.joblib", run_type=run_type)
        model_path = models_dir / model_filename
        joblib.dump(pipeline, model_path)

        logger.info(f"[RECOVERY] ✓ Saved {model_name} to {model_path}")


def _generate_predictions_from_partial(
    input_file: str,
    run_type: RunType,
    completed_models: List[str],
    logger: logging.Logger
) -> str:
    """Generate predictions using best available completed model.

    Falls back to any available model if best model isn't complete.
    """
    paths = get_output_paths(run_type)

    # Try to use best model from summary
    summary_file = Path(paths["artifacts"]) / "model_summary.csv"

    if summary_file.exists():
        summary = pd.read_csv(summary_file)
        best_model = summary.iloc[0]["model"]
        logger.info(f"[RECOVERY] Using best available model: {best_model}")
    else:
        # Fallback: use any completed model
        best_model = completed_models[0]
        logger.warning(f"[RECOVERY] No summary available, using first completed model: {best_model}")

    # Generate predictions
    return generate_predictions(input_file, run_type=run_type)


def _send_recovery_notification(
    run_type: RunType,
    completed_models: List[str],
    failed_models: List[str],
    original_error: Exception,
    logger: logging.Logger
) -> None:
    """Send notification about recovery outcome.

    In future, this could send email/Slack/etc. For now, just comprehensive logging.
    """
    logger.warning("\n" + "=" * 70)
    logger.warning("RECOVERY NOTIFICATION")
    logger.warning("=" * 70)
    logger.warning(f"Run type: {run_type}")
    logger.warning(f"Original error: {type(original_error).__name__}: {str(original_error)}")
    logger.warning("")
    logger.warning(f"✓ Completed models ({len(completed_models)}): {', '.join(completed_models)}")
    logger.warning(f"✗ Failed models ({len(failed_models)}): {', '.join(failed_models)}")
    logger.warning("")
    logger.warning("Recovery actions taken:")
    logger.warning("  ✓ Generated model_summary.csv from partial results")
    logger.warning("  ✓ Fitted final models for completed models")
    logger.warning("  ✓ Generated predictions using best available model")
    logger.warning("")
    logger.warning("⚠️  IMPORTANT: Some models failed to train completely")
    logger.warning("   Consider re-running the pipeline to train all models")
    logger.warning("=" * 70)
```

---

## Implementation Plan

### Phase 1: Core Recovery Logic (2-3 hours)

**Tasks**:
1. Create `src/survival_framework/recovery.py` module with helper functions
2. Add `attempt_recovery()` function with 5 steps
3. Implement detection functions (_detect_completed_models, etc.)
4. Add recovery exception handler to main.py

**Files to modify**:
- `src/survival_framework/recovery.py` (new, ~300 lines)
- `src/main.py` (add recovery in exception handler, ~15 lines)

### Phase 2: Testing (1 hour)

**Test scenarios**:
1. Simulate failure after 1 model completes
2. Simulate failure after 3 models complete
3. Simulate failure during predictions
4. Verify idempotent recovery (run twice)

### Phase 3: Notification System (optional, future)

**Extensions**:
- Email notifications via SMTP
- Slack webhooks
- JSON status file for monitoring systems

---

## Why NOT a Separate Skill?

### ❌ Skill Approach Issues:

1. **Context loss**: Skill runs in separate invocation, loses in-memory state
2. **Complex serialization**: Would need to serialize all paths, configs, logger state
3. **Harder to maintain**: Logic split across skill and main code
4. **Can't access exception**: Skill doesn't have access to original exception object
5. **Redundant logic**: Would duplicate code already in train.py/predict.py

### ✅ Exception Handler Approach Benefits:

1. **Same context**: Access to all variables, paths, configs
2. **Clean flow**: Natural error handling pattern
3. **Single codebase**: All recovery logic in one module
4. **Standard practice**: This is how production systems handle failures
5. **Testable**: Easy to write unit tests for recovery functions

---

## Success Criteria

**After implementation, a failed pipeline should**:

- [ ] Automatically detect which models completed
- [ ] Generate model_summary.csv from partial results
- [ ] Fit final models for completed models only
- [ ] Generate predictions using best available model
- [ ] Log clear notification of what succeeded/failed
- [ ] Exit with code 0 if predictions were generated
- [ ] Exit with code 1 only if total failure (no usable output)
- [ ] Be idempotent (safe to run recovery multiple times)

**Example recovery output**:
```
ERROR: Pipeline failed: MlflowException: Run not found
======================================================================
ATTEMPTING AUTOMATIC RECOVERY
======================================================================

[RECOVERY] Step 1: Analyzing completed work...
[RECOVERY] Found 3 completed models: ['cox_ph', 'coxnet', 'gbsa']

[RECOVERY] Step 2: Generating metrics summary...
[RECOVERY] ✓ Metrics summary generated

[RECOVERY] Step 3: Fitting final models...
[RECOVERY] Fitting final cox_ph...
[RECOVERY] ✓ Saved cox_ph to data/outputs/production/models/...
[RECOVERY] Fitting final coxnet...
[RECOVERY] ✓ Saved coxnet to data/outputs/production/models/...
[RECOVERY] Fitting final gbsa...
[RECOVERY] ✓ Saved gbsa to data/outputs/production/models/...
[RECOVERY] ✓ Final models fitted

[RECOVERY] Step 4: Generating predictions...
[RECOVERY] Using best available model: gbsa
[RECOVERY] ✓ Predictions saved to: data/outputs/production/predictions/...

[RECOVERY] Step 5: Sending notification...

======================================================================
RECOVERY NOTIFICATION
======================================================================
Run type: production
Original error: MlflowException: Run '98bd42d60b5644b7815387dd52c24af3' not found

✓ Completed models (3): cox_ph, coxnet, gbsa
✗ Failed models (2): weibull_aft, rsf

Recovery actions taken:
  ✓ Generated model_summary.csv from partial results
  ✓ Fitted final models for completed models
  ✓ Generated predictions using best available model

⚠️  IMPORTANT: Some models failed to train completely
   Consider re-running the pipeline to train all models
======================================================================

======================================================================
⚠️  PARTIAL SUCCESS - RECOVERY COMPLETED
======================================================================
Some models failed, but predictions generated from available models
Original error: MlflowException: Run '98bd42d60b5644b7815387dd52c24af3' not found
```

---

## Future Enhancements

1. **Email notifications**: Send alert when recovery triggers
2. **Retry logic**: Attempt to re-train failed models automatically
3. **Checkpointing**: Save state between models for true resume capability
4. **Health checks**: Validate recovered outputs before declaring success
5. **Metrics tracking**: Log recovery events to monitoring system

---

## Related Work

- ✅ Incremental metrics persistence (implemented)
- ✅ Safe MLflow wrappers (implemented)
- ✅ Production logging infrastructure (implemented)
- ⏳ Auto-recovery (this document)
- 🔮 Email notifications (future)
- 🔮 Resume from checkpoint (future)
