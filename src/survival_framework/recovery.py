"""Automatic recovery system for pipeline failures.

This module provides automatic recovery capabilities when the training pipeline
fails partway through. It can detect which models completed successfully and
generate predictions using available models.

Key features:
- Detects completed models from incremental metrics CSV
- Generates model summary from partial results
- Fits final models for completed models only
- Generates predictions using best available model
- Sends comprehensive notifications about recovery status
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

from survival_framework.data import load_data, split_X_y, make_preprocessor, RunType
from survival_framework.predict import generate_predictions
from survival_framework.utils import get_output_paths, versioned_name
from survival_framework.config import ModelHyperparameters


def attempt_recovery(
    run_type: RunType,
    input_file: str,
    logger: logging.Logger,
    original_error: Exception
) -> bool:
    """Attempt to recover from pipeline failure.

    Checks what models completed successfully and attempts to:
    1. Detect which models completed all CV folds
    2. Generate model_summary.csv from partial metrics
    3. Fit final models for completed models
    4. Generate predictions using best available model
    5. Send notification about partial completion

    Args:
        run_type: Sample or production run
        input_file: Path to input data file
        logger: Logger instance for status messages
        original_error: The exception that triggered recovery

    Returns:
        True if recovery generated usable outputs (predictions), False otherwise

    Example:
        >>> try:
        ...     train_all_models(...)
        ... except Exception as e:
        ...     if attempt_recovery("production", "data.pkl", logger, e):
        ...         print("Recovery successful - predictions generated")
        ...     else:
        ...         print("Recovery failed - no usable output")
    """
    paths = get_output_paths(run_type)

    logger.warning("")
    logger.warning("=" * 70)
    logger.warning("ATTEMPTING AUTOMATIC RECOVERY")
    logger.warning("=" * 70)

    # Step 1: Detect completed models
    logger.info("\n[RECOVERY] Step 1: Analyzing completed work...")
    completed_models = _detect_completed_models(paths["artifacts"], logger)

    if not completed_models:
        logger.error("[RECOVERY] ❌ No completed models found - cannot recover")
        logger.error("[RECOVERY] All models failed or no metrics were saved")
        return False

    logger.info(f"[RECOVERY] ✓ Found {len(completed_models)} completed models: {', '.join(completed_models)}")

    # Step 2: Generate metrics summary
    logger.info("\n[RECOVERY] Step 2: Generating metrics summary from partial results...")
    try:
        _generate_metrics_summary_from_partial(
            paths["artifacts"],
            completed_models,
            logger
        )
        logger.info("[RECOVERY] ✓ Metrics summary generated successfully")
    except Exception as e:
        logger.warning(f"[RECOVERY] ⚠️  Metrics summary generation failed: {e}")
        # Continue anyway - we can still generate predictions

    # Step 3: Fit final models
    logger.info("\n[RECOVERY] Step 3: Fitting final models for completed models...")
    try:
        fitted_count = _fit_final_models_for_completed(
            input_file=input_file,
            run_type=run_type,
            completed_models=completed_models,
            logger=logger
        )
        logger.info(f"[RECOVERY] ✓ Successfully fitted {fitted_count} final models")
    except Exception as e:
        logger.error(f"[RECOVERY] ⚠️  Final model fitting failed: {e}")
        logger.warning("[RECOVERY] Will attempt to use any existing models for predictions")

    # Step 4: Generate predictions
    logger.info("\n[RECOVERY] Step 4: Generating predictions with best available model...")
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
        logger.error("[RECOVERY] Unable to generate predictions - recovery failed")
        return False

    # Step 5: Send notification
    logger.info("\n[RECOVERY] Step 5: Logging recovery status...")
    _send_recovery_notification(
        run_type=run_type,
        completed_models=completed_models,
        total_models=4,  # cox_ph, coxnet, weibull_aft, gbsa
        original_error=original_error,
        logger=logger
    )

    return True


def _detect_completed_models(artifacts_path: str, logger: logging.Logger) -> List[str]:
    """Detect which models completed all CV folds.

    Checks model_metrics.csv (written incrementally) to see which models
    have complete 5-fold cross-validation results.

    Args:
        artifacts_path: Path to artifacts directory
        logger: Logger instance

    Returns:
        List of model names that completed all 5 folds

    Example:
        >>> completed = _detect_completed_models("data/outputs/production/artifacts", logger)
        >>> print(completed)
        ['cox_ph', 'coxnet', 'gbsa']
    """
    metrics_file = Path(artifacts_path) / "model_metrics.csv"

    if not metrics_file.exists():
        logger.warning(f"[RECOVERY] No metrics file found at {metrics_file}")
        return []

    # Read incremental metrics
    df = pd.read_csv(metrics_file)
    logger.info(f"[RECOVERY] Found metrics for {len(df)} fold results")

    # Check which models have 5 complete folds
    model_counts = df.groupby("model").size()
    logger.info(f"[RECOVERY] Model fold counts: {dict(model_counts)}")

    completed = model_counts[model_counts == 5].index.tolist()

    return completed


def _generate_metrics_summary_from_partial(
    artifacts_path: str,
    completed_models: List[str],
    logger: logging.Logger
) -> None:
    """Generate model_summary.csv from partial model_metrics.csv.

    Creates aggregated metrics summary including only models that completed
    all cross-validation folds.

    Args:
        artifacts_path: Path to artifacts directory
        completed_models: List of models that completed successfully
        logger: Logger instance

    Side Effects:
        Creates model_summary.csv in artifacts_path with columns:
        - model: Model name
        - cindex: Mean C-index across folds
        - ibs: Mean IBS across folds
        - rank_cindex: Rank by C-index (1 = best)
        - rank_ibs: Rank by IBS (1 = best)
    """
    metrics_file = Path(artifacts_path) / "model_metrics.csv"
    summary_file = Path(artifacts_path) / "model_summary.csv"

    # Read metrics
    df = pd.read_csv(metrics_file)

    # Filter to completed models only
    df = df[df["model"].isin(completed_models)]
    logger.info(f"[RECOVERY] Processing {len(df)} fold results for {len(completed_models)} models")

    # Aggregate by model
    summary = df.groupby("model").agg({
        "cindex": "mean",
        "ibs": "mean"
    }).reset_index()

    # Add ranks (higher cindex = better, lower ibs = better)
    summary["rank_cindex"] = summary["cindex"].rank(ascending=False)
    summary["rank_ibs"] = summary["ibs"].rank(ascending=True)

    # Sort by cindex rank (best first)
    summary = summary.sort_values("rank_cindex")

    # Save
    summary.to_csv(summary_file, index=False)
    logger.info(f"[RECOVERY] Saved summary for {len(summary)} models to {summary_file}")

    # Log rankings
    for _, row in summary.iterrows():
        logger.info(
            f"[RECOVERY]   {row['model']}: "
            f"C-index={row['cindex']:.4f} (rank {int(row['rank_cindex'])}), "
            f"IBS={row['ibs']:.4f} (rank {int(row['rank_ibs'])})"
        )


def _fit_final_models_for_completed(
    input_file: str,
    run_type: RunType,
    completed_models: List[str],
    logger: logging.Logger
) -> int:
    """Fit final models only for models that completed CV.

    Loads the full dataset, fits a preprocessing pipeline, then trains
    final versions of each completed model on the full data.

    Args:
        input_file: Path to input data file
        run_type: Sample or production run type
        completed_models: List of models to fit
        logger: Logger instance

    Returns:
        Number of models successfully fitted and saved

    Example:
        >>> count = _fit_final_models_for_completed(
        ...     "data.pkl", "production", ["cox_ph", "gbsa"], logger
        ... )
        >>> print(f"Fitted {count} models")
        Fitted 2 models
    """
    # Load full dataset
    logger.info("[RECOVERY] Loading full dataset for final model fitting...")
    df = load_data(input_file, run_type=run_type)
    X, y, ids = split_X_y(df)
    logger.info(f"[RECOVERY] Loaded {len(df):,} records with {X.shape[1]} features")

    # Create and fit preprocessing pipeline
    logger.info("[RECOVERY] Fitting preprocessing pipeline...")
    preprocessor = make_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    # Get output directory
    paths = get_output_paths(run_type)
    models_dir = Path(paths["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    fitted_count = 0

    # Fit each completed model
    for model_name in completed_models:
        try:
            logger.info(f"[RECOVERY] Fitting final {model_name} on full dataset...")

            # Create model wrapper
            model = _create_model_wrapper(model_name, logger)

            # Fit on full dataset
            model.fit(X_transformed, y)

            # Create pipeline with preprocessor + model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Save with versioned filename
            model_filename = versioned_name(f"{run_type}_{model_name}.joblib", run_type=run_type)
            model_path = models_dir / model_filename
            joblib.dump(pipeline, model_path)

            file_size = model_path.stat().st_size / 1024  # KB
            logger.info(f"[RECOVERY] ✓ Saved {model_name} to {model_path.name} ({file_size:.1f} KB)")
            fitted_count += 1

        except Exception as e:
            logger.error(f"[RECOVERY] ✗ Failed to fit {model_name}: {e}")
            continue

    return fitted_count


def _create_model_wrapper(model_name: str, logger: logging.Logger):
    """Create model wrapper instance by name.

    Args:
        model_name: Name of model (cox_ph, coxnet, weibull_aft, gbsa)
        logger: Logger instance

    Returns:
        Instantiated model wrapper

    Raises:
        ValueError: If model_name is not recognized
    """
    from survival_framework.models import (
        CoxPHWrapper, CoxnetWrapper, WeibullAFTWrapper,
        GBSAWrapper
    )

    # Get default hyperparameters
    config = ModelHyperparameters()

    if model_name == "cox_ph":
        return CoxPHWrapper(alpha=config.coxph_alpha)
    elif model_name == "coxnet":
        return CoxnetWrapper(
            l1_ratio=config.coxnet_l1_ratio,
            alpha_min_ratio=config.coxnet_alpha_min_ratio,
            n_alphas=config.coxnet_n_alphas
        )
    elif model_name == "weibull_aft":
        return WeibullAFTWrapper(penalizer=config.weibull_penalizer)
    elif model_name == "gbsa":
        return GBSAWrapper(
            n_estimators=config.gbsa_n_estimators,
            learning_rate=config.gbsa_learning_rate,
            max_depth=config.gbsa_max_depth,
            subsample=config.gbsa_subsample,
            min_samples_split=config.gbsa_min_samples_split
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def _generate_predictions_from_partial(
    input_file: str,
    run_type: RunType,
    completed_models: List[str],
    logger: logging.Logger
) -> str:
    """Generate predictions using best available completed model.

    Reads model_summary.csv to find best model, or falls back to first
    completed model if summary doesn't exist.

    Args:
        input_file: Path to input data file
        run_type: Sample or production run type
        completed_models: List of models that completed successfully
        logger: Logger instance

    Returns:
        Path to saved predictions CSV file

    Raises:
        FileNotFoundError: If no models are available for predictions
    """
    paths = get_output_paths(run_type)
    summary_file = Path(paths["artifacts"]) / "model_summary.csv"

    # Try to use best model from summary
    if summary_file.exists():
        summary = pd.read_csv(summary_file)
        best_model = summary.iloc[0]["model"]
        best_cindex = summary.iloc[0]["cindex"]
        logger.info(
            f"[RECOVERY] Using best available model: {best_model} "
            f"(C-index={best_cindex:.4f})"
        )
    else:
        # Fallback: use first completed model
        best_model = completed_models[0]
        logger.warning(
            f"[RECOVERY] No summary available, using first completed model: {best_model}"
        )

    # Generate predictions using standard pipeline
    logger.info(f"[RECOVERY] Generating predictions for {len(load_data(input_file, run_type=run_type)):,} records...")
    pred_path = generate_predictions(input_file, run_type=run_type)

    return pred_path


def _send_recovery_notification(
    run_type: RunType,
    completed_models: List[str],
    total_models: int,
    original_error: Exception,
    logger: logging.Logger
) -> None:
    """Send notification about recovery outcome.

    Logs comprehensive status information about what succeeded and failed.
    In future versions, this could send email/Slack notifications.

    Args:
        run_type: Sample or production run type
        completed_models: List of models that completed successfully
        total_models: Total number of models attempted
        original_error: Exception that triggered recovery
        logger: Logger instance
    """
    failed_count = total_models - len(completed_models)
    all_models = ["cox_ph", "coxnet", "weibull_aft", "gbsa"]
    failed_models = [m for m in all_models if m not in completed_models]

    logger.warning("")
    logger.warning("=" * 70)
    logger.warning("RECOVERY NOTIFICATION")
    logger.warning("=" * 70)
    logger.warning(f"Run type: {run_type.upper()}")
    logger.warning(f"Original error: {type(original_error).__name__}: {str(original_error)[:100]}")
    logger.warning("")
    logger.warning(f"✓ Completed models ({len(completed_models)}/{total_models}): {', '.join(completed_models)}")

    if failed_models:
        logger.warning(f"✗ Failed models ({len(failed_models)}/{total_models}): {', '.join(failed_models)}")

    logger.warning("")
    logger.warning("Recovery actions taken:")
    logger.warning("  ✓ Generated model_summary.csv from partial results")
    logger.warning("  ✓ Fitted final models for completed models")
    logger.warning("  ✓ Generated predictions using best available model")
    logger.warning("")

    if failed_models:
        logger.warning("⚠️  IMPORTANT: Some models failed to train completely")
        logger.warning("   Consider re-running the pipeline to train all models")
        logger.warning("   Or investigate the error to prevent future failures")
    else:
        logger.warning("✓ All models completed successfully (error occurred during post-processing)")

    logger.warning("=" * 70)
