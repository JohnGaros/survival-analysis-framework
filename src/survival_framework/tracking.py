from __future__ import annotations
import os
import json
import logging
from typing import Dict, Any, Optional
import mlflow
import mlflow.exceptions


def start_run(run_name: str, tags: Dict[str, str] | None = None):
    """Start MLflow tracking run under survival_framework experiment.

    Initializes an MLflow run for experiment tracking, setting the experiment
    name to "survival_framework" and optionally adding custom tags.

    Args:
        run_name: Name identifier for this run
        tags: Optional dictionary of key-value tags to attach to the run

    Returns:
        Active MLflow run context manager

    Example:
        >>> with start_run("model_comparison_v1", tags={"author": "data_team"}):
        ...     # Training and logging code here
        ...     pass
    """
    mlflow.set_experiment("survival_framework")
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params(params: Dict[str, Any]):
    """Log parameters to current MLflow run.

    Records hyperparameters and configuration values. Automatically converts
    non-serializable values to strings.

    Args:
        params: Dictionary of parameter names and values to log

    Example:
        >>> log_params({"n_splits": 5, "random_state": 42, "model": "cox_ph"})
    """
    for k, v in params.items():
        try:
            mlflow.log_param(k, v)
        except Exception:
            mlflow.log_param(k, str(v))


def log_metrics(metrics: Dict[str, float], step: int | None = None):
    """Log metrics to current MLflow run.

    Records performance metrics such as C-index, IBS, and AUC. Optionally
    associates metrics with a step number for tracking over iterations.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., fold index, epoch). Defaults to None

    Example:
        >>> log_metrics({"cindex": 0.742, "ibs": 0.152}, step=0)
        >>> log_metrics({"cindex": 0.738, "ibs": 0.148}, step=1)
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str):
    """Log file artifact to current MLflow run.

    Uploads a file (e.g., model, plot, CSV) to MLflow for later retrieval.
    Only logs if the file exists.

    Args:
        path: File path to log as artifact

    Example:
        >>> log_artifact("artifacts/model_summary.csv")
        >>> log_artifact("artifacts/ph_flags.csv")
    """
    if os.path.exists(path):
        mlflow.log_artifact(path)


def log_dict(name: str, d: Dict[str, Any]):
    """Log dictionary as JSON artifact to current MLflow run.

    Serializes dictionary to JSON file and uploads as MLflow artifact.
    Useful for logging configuration objects or structured results.

    Args:
        name: Base name for the JSON file (without extension)
        d: Dictionary to serialize and log

    Example:
        >>> config = {"model": "cox_ph", "params": {"alpha": 0.5}}
        >>> log_dict("model_config", config)
    """
    tmp = f"/tmp/{name}.json"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    mlflow.log_artifact(tmp)


# ============================================================================
# Safe MLflow Wrappers with Graceful Degradation
# ============================================================================


def safe_log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Log metrics to MLflow with error handling.

    Attempts to log metrics to MLflow. If MLflow fails, logs a warning but
    continues execution. Metrics should still be persisted to CSV files.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., fold index)
        logger: Optional logger for warnings

    Returns:
        True if logging succeeded, False if it failed

    Example:
        >>> success = safe_log_metrics({"cindex": 0.742}, step=0, logger=logger)
        >>> if not success:
        ...     print("MLflow unavailable, metrics saved to CSV only")
    """
    try:
        mlflow.log_metrics(metrics, step=step)
        return True
    except mlflow.exceptions.MlflowException as e:
        if logger:
            logger.warning(
                f"MLflow metrics logging failed: {e}",
                extra={"category": "mlflow_error"}
            )
        return False
    except Exception as e:
        if logger:
            logger.error(
                f"Unexpected error in MLflow metrics logging: {e}",
                extra={"category": "mlflow_error"}
            )
        return False


def safe_log_params(
    params: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> bool:
    """Log parameters to MLflow with error handling.

    Attempts to log parameters to MLflow. If MLflow fails, logs a warning but
    continues execution.

    Args:
        params: Dictionary of parameter names and values
        logger: Optional logger for warnings

    Returns:
        True if logging succeeded, False if it failed

    Example:
        >>> success = safe_log_params({"n_splits": 5}, logger=logger)
    """
    try:
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                mlflow.log_param(k, str(v))
        return True
    except mlflow.exceptions.MlflowException as e:
        if logger:
            logger.warning(
                f"MLflow params logging failed: {e}",
                extra={"category": "mlflow_error"}
            )
        return False
    except Exception as e:
        if logger:
            logger.error(
                f"Unexpected error in MLflow params logging: {e}",
                extra={"category": "mlflow_error"}
            )
        return False


def safe_log_artifact(
    path: str,
    logger: Optional[logging.Logger] = None
) -> bool:
    """Log artifact to MLflow with error handling.

    Attempts to log artifact to MLflow. If MLflow fails, logs a warning but
    continues execution.

    Args:
        path: File path to log as artifact
        logger: Optional logger for warnings

    Returns:
        True if logging succeeded, False if it failed

    Example:
        >>> success = safe_log_artifact("artifacts/model_summary.csv", logger=logger)
    """
    if not os.path.exists(path):
        if logger:
            logger.warning(f"Artifact not found, skipping: {path}")
        return False

    try:
        mlflow.log_artifact(path)
        return True
    except mlflow.exceptions.MlflowException as e:
        if logger:
            logger.warning(
                f"MLflow artifact logging failed for {path}: {e}",
                extra={"category": "mlflow_error"}
            )
        return False
    except Exception as e:
        if logger:
            logger.error(
                f"Unexpected error in MLflow artifact logging for {path}: {e}",
                extra={"category": "mlflow_error"}
            )
        return False