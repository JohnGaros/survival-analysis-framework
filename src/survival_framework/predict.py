"""Prediction module for generating survival probabilities from trained models.

This module provides functionality to load the best-performing model and generate
survival probability predictions for new data.
"""
from __future__ import annotations
import os
import glob
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy import integrate

from survival_framework.data import split_X_y, load_data, RunType
from survival_framework.utils import ensure_dir, default_time_grid, get_output_paths


def calculate_expected_survival_time(
    pipeline,
    X,
    max_time: float = None,
    n_points: int = 200
) -> np.ndarray:
    """Calculate expected survival time for each sample.

    Computes the restricted mean survival time (RMST), which is the area under
    the survival curve up to a maximum time point. This represents the average
    time until the event (termination) occurs, restricted to the time window.

    Args:
        pipeline: Fitted survival model pipeline
        X: Feature matrix with shape (n_samples, n_features)
        max_time: Maximum time horizon for integration (months). If None, uses
            the maximum observed time from the training data. Defaults to None
        n_points: Number of points to use for numerical integration. Defaults to 200

    Returns:
        Array with shape (n_samples,) containing expected survival time in months
        for each sample

    Example:
        >>> expected_months = calculate_expected_survival_time(pipeline, X_test)
        >>> print(f"Average expected survival: {expected_months.mean():.1f} months")
        Average expected survival: 42.3 months

    Notes:
        - Expected time is calculated as integral of S(t) from 0 to max_time
        - Higher expected time = lower risk = longer expected survival
        - If survival curve doesn't drop to 0 within max_time, this gives
          a lower bound on true expected survival time
        - Uses trapezoidal integration for numerical stability
    """
    # Get maximum time from training data if not specified
    if max_time is None:
        # Extract from the underlying model's unique times
        model = pipeline.named_steps["model"]
        if hasattr(model, 'estimator_') and hasattr(model.estimator_, 'unique_times_'):
            max_time = float(model.estimator_.unique_times_.max())
        else:
            # Fallback to a reasonable default
            max_time = 100.0

    # Create fine-grained time grid for integration
    times = np.linspace(0, max_time, n_points)

    # Get survival probabilities at all time points
    surv_probs = pipeline.predict_survival_function(X, times=times)

    # Calculate area under curve (expected survival time) using trapezoidal rule
    # For each sample, integrate S(t) over time
    expected_times = np.array([
        integrate.trapezoid(surv_probs[i, :], times)
        for i in range(surv_probs.shape[0])
    ])

    return expected_times


def load_best_model(artifacts_dir: str = "artifacts", models_dir: str = "models") -> Tuple[str, object]:
    """Load the best-performing model based on cross-validation results.

    Reads the model summary CSV to identify the best model (highest C-index),
    then loads the most recent saved version of that model.

    Args:
        artifacts_dir: Directory containing model_summary.csv with rankings
        models_dir: Directory containing saved model joblib files

    Returns:
        Tuple containing:
        - model_name: Name of the best model (e.g., "cox_ph", "rsf")
        - pipeline: Loaded sklearn Pipeline with fitted preprocessor and model

    Raises:
        FileNotFoundError: If model_summary.csv or model files are not found
        ValueError: If no models are available in models_dir

    Example:
        >>> model_name, pipeline = load_best_model()
        >>> print(f"Best model: {model_name}")
        Best model: rsf
    """
    summary_path = os.path.join(artifacts_dir, "model_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Model summary not found at {summary_path}. "
            "Please run training first using train_all_models()."
        )

    # Read summary and get best model (first row after sorting by rank_cindex)
    summary = pd.read_csv(summary_path)
    best_model_name = summary.iloc[0]["model"]

    # Find most recent saved model file for best model
    # Pattern matches both legacy (model_name.joblib_*) and new (runtype_model_name.joblib_*) formats
    pattern = os.path.join(models_dir, f"*{best_model_name}.joblib_*")
    model_files = glob.glob(pattern)

    if not model_files:
        raise ValueError(
            f"No saved models found for {best_model_name} in {models_dir}. "
            "Please run training first."
        )

    # Get most recent file (sorted by timestamp in filename)
    latest_model_path = sorted(model_files)[-1]
    pipeline = joblib.load(latest_model_path)

    return best_model_name, pipeline


def generate_predictions(
    file_path: str,
    run_type: RunType = "sample",
    time_horizons: List[int] = None,
) -> str:
    """Generate survival probability predictions for all records in input file.

    Loads the best-performing model and generates survival probability predictions
    at specified time horizons for all records. Saves results to CSV with one row
    per account_entities_key.

    Args:
        file_path: Path to input file (CSV or pickle) with same schema as training data
        run_type: Type of run - "sample" for development, "production" for full data.
            Determines which model and output directory to use.
        time_horizons: List of time points (months) for survival predictions.
            Defaults to [3, 6, 12, 18, 24, 36] if None

    Returns:
        Path to saved predictions CSV file

    Side Effects:
        Creates output directory if it doesn't exist and writes predictions CSV with columns:
        - account_entities_key: Account identifier
        - model_name: Name of the model used
        - expected_survival_months: Expected survival time (restricted mean)
        - survival_prob_Xm: Survival probability at X months (one column per horizon)

    Example:
        >>> # Sample run
        >>> pred_path = generate_predictions(
        ...     "data/inputs/sample/data.csv",
        ...     run_type="sample"
        ... )
        >>> print(f"Predictions saved to: {pred_path}")
        Predictions saved to: data/outputs/sample/predictions/survival_predictions_sample_20251023_143052.csv

        >>> # Production run
        >>> pred_path = generate_predictions(
        ...     "data/inputs/production/data.pkl",
        ...     run_type="production"
        ... )

    Notes:
        - Uses the best model based on C-index from cross-validation
        - Survival probability at time t: P(survival time > t)
        - Higher survival probability = lower risk of termination
        - Outputs always saved as CSV regardless of input format
    """
    if time_horizons is None:
        time_horizons = [3, 6, 12, 18, 24, 36]

    # Get paths for this run type
    paths = get_output_paths(run_type)

    # Load best model from correct location
    model_name, pipeline = load_best_model(paths["artifacts"], paths["models"])
    print(f"Using best model: {model_name} ({run_type})")

    # Load data (supports CSV and pickle)
    df = load_data(file_path, run_type=run_type)
    X, y, ids = split_X_y(df, dropna=False)  # Don't drop NA - imputation handles it

    # Generate predictions for all time horizons at once
    results = {"account_entities_key": ids.values, "model_name": model_name}

    # Calculate expected survival time (restricted mean survival time)
    print("Calculating expected survival time...")
    expected_months = calculate_expected_survival_time(pipeline, X)
    results["expected_survival_months"] = expected_months

    # Predict survival functions at all time horizons
    surv_funcs = pipeline.predict_survival_function(X, times=time_horizons)

    # surv_funcs has shape (n_samples, n_times)
    for i, t in enumerate(time_horizons):
        results[f"survival_prob_{t}m"] = surv_funcs[:, i]

    # Create output DataFrame
    pred_df = pd.DataFrame(results)

    # Save to CSV with run_type in filename
    output_filename = f"survival_predictions_{run_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = os.path.join(paths["predictions"], output_filename)
    pred_df.to_csv(output_path, index=False)

    print(f"\n[{run_type.upper()}] Predictions saved to: {output_path}")
    print(f"Generated predictions for {len(pred_df):,} accounts at {len(time_horizons)} time horizons")

    return output_path
