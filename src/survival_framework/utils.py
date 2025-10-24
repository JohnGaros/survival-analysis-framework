from __future__ import annotations
import os
import datetime as dt
import numpy as np
import pandas as pd
from typing import Literal

# Run type for distinguishing sample vs production runs
RunType = Literal["sample", "production"]


def ensure_dir(path: str):
    """Create directory if it doesn't exist.

    Creates the specified directory path, including any necessary parent
    directories. Does nothing if the directory already exists.

    Args:
        path: Directory path to create

    Example:
        >>> ensure_dir("artifacts/models")
        >>> ensure_dir("results/fold_predictions")
    """
    os.makedirs(path, exist_ok=True)


def default_time_grid(y_struct, n: int = 50, y_test=None) -> np.ndarray:
    """Generate default time grid for survival function evaluation.

    Creates evenly spaced time points from the minimum to maximum observed times,
    constrained to be within the test set's follow-up range if provided.
    Used for evaluating survival functions at consistent time horizons.

    Args:
        y_struct: Structured array with dtype=[('event', bool), ('time', float)]
                  Training set or full dataset for determining time range
        n: Number of time points to generate. Defaults to 50
        y_test: Optional test set structured array. If provided, ensures all time
                points are within the test set's observed time range to avoid
                extrapolation errors in metrics like IBS and AUC.

    Returns:
        Array of time points with shape (n,), constrained to valid range

    Example:
        >>> # For full dataset predictions
        >>> times = default_time_grid(y_train, n=100)

        >>> # For cross-validation (constrained to test fold range)
        >>> times = default_time_grid(y_train, n=50, y_test=y_test)
        >>> print(f"Time range: {times[0]:.1f} to {times[-1]:.1f}")
        Time range: 2.5 to 48.3

    Notes:
        - When y_test is provided, time grid is constrained to [min(y_test), max(y_test)]
        - This prevents "all times must be within follow-up time" errors
        - For metrics like IBS and time-dependent AUC, times must be within test data range
    """
    # Get reasonable time range from training data
    t_min = float(np.percentile(y_struct["time"], 5))  # 5th percentile to avoid extreme lows
    t_max = float(np.percentile(y_struct["time"], 95))  # 95th percentile

    # If test set provided, constrain to its observed range
    if y_test is not None:
        test_min = float(y_test["time"].min())
        test_max = float(y_test["time"].max())

        # Constrain to intersection of train and test ranges
        # Add small buffer to avoid boundary issues (exclusive upper bound in scikit-survival)
        t_min = max(t_min, test_min + 0.1)
        t_max = min(t_max, test_max - 0.1)

        # Ensure valid range
        if t_min >= t_max:
            # Fallback to safe middle range
            t_min = test_min + 0.1
            t_max = test_max - 0.1

    return np.linspace(t_min, t_max, n)


def save_model_metrics(df: pd.DataFrame, outdir: str):
    """Save model metrics DataFrame to CSV file.

    Writes cross-validation metrics to disk in the specified output directory.
    Creates directory if it doesn't exist.

    Args:
        df: DataFrame containing model performance metrics
        outdir: Output directory path

    Returns:
        Full path to the saved CSV file

    Example:
        >>> metrics_df = pd.DataFrame({
        ...     "model": ["cox_ph", "rsf"],
        ...     "fold": [0, 0],
        ...     "cindex": [0.74, 0.76]
        ... })
        >>> path = save_model_metrics(metrics_df, "artifacts")
        >>> print(f"Metrics saved to: {path}")
        Metrics saved to: artifacts/model_metrics.csv
    """
    ensure_dir(outdir)
    path = os.path.join(outdir, "model_metrics.csv")
    df.to_csv(path, index=False)
    return path


def versioned_name(base: str, run_type: RunType = None) -> str:
    """Generate timestamped filename for versioning.

    Appends current timestamp to base name for creating unique versioned
    filenames. Optionally includes run_type prefix for clarity.

    Args:
        base: Base filename without extension
        run_type: Optional run type ("sample" or "production") to prefix filename

    Returns:
        Versioned name in format "[runtype_]base_YYYYMMDD_HHMMSS"

    Example:
        >>> name = versioned_name("cox_ph")
        >>> print(name)
        cox_ph_20250123_143052

        >>> name = versioned_name("cox_ph", run_type="sample")
        >>> print(name)
        sample_cox_ph_20250123_143052
    """
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_type:
        return f"{run_type}_{base}_{ts}"
    return f"{base}_{ts}"


def get_output_paths(run_type: RunType = "sample") -> dict:
    """Get standardized output directory paths for a given run type.

    Creates and returns paths for all output directories, ensuring they
    are properly organized by run type (sample vs production).

    Args:
        run_type: Type of run - "sample" for development, "production" for full data

    Returns:
        Dictionary with keys:
        - base_dir: Root output directory for this run type
        - predictions: Directory for prediction CSVs
        - artifacts: Directory for training artifacts (metrics, PH flags)
        - models: Directory for saved model files
        - mlruns: Directory for MLflow tracking

    Example:
        >>> paths = get_output_paths("sample")
        >>> print(paths["predictions"])
        data/outputs/sample/predictions

        >>> paths = get_output_paths("production")
        >>> print(paths["models"])
        data/outputs/production/models

    Notes:
        - All paths are created if they don't exist
        - Maintains separation between sample and production runs
        - Used by training and prediction pipelines
    """
    base_dir = f"data/outputs/{run_type}"

    paths = {
        "base_dir": base_dir,
        "predictions": os.path.join(base_dir, "predictions"),
        "artifacts": os.path.join(base_dir, "artifacts"),
        "models": os.path.join(base_dir, "models"),
        "mlruns": os.path.join(base_dir, "mlruns"),
    }

    # Ensure all directories exist
    for path in paths.values():
        ensure_dir(path)

    return paths