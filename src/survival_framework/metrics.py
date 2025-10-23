from __future__ import annotations
from typing import Dict, Iterable, Tuple
import numpy as np
from sksurv.metrics import (
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)


def compute_cindex(y_train, y_test, risk_scores) -> float:
    """Calculate Harrell's concordance index using IPCW adjustment.

    Computes concordance index (C-index) for survival predictions with
    inverse probability of censoring weighting (IPCW) to handle right-censored data.
    The C-index measures the model's ability to correctly order pairs of samples
    by their survival times.

    Args:
        y_train: Structured array from training set with dtype=[('event', bool), ('time', float)]
            Used to estimate censoring distribution
        y_test: Structured array from test set with dtype=[('event', bool), ('time', float)]
        risk_scores: Array of shape (n_test,) with predicted risk scores.
            Higher values indicate higher risk (lower survival probability)

    Returns:
        Concordance index between 0.5 (random) and 1.0 (perfect discrimination)

    Example:
        >>> cindex = compute_cindex(y_train, y_test, risk_scores)
        >>> print(f"C-index: {cindex:.3f}")
        C-index: 0.742
    """
    result = concordance_index_ipcw(y_train, y_test, risk_scores)
    cindex = result[0]  # concordance_index_ipcw returns tuple: (cindex, concordant, discordant, tied_risk, tied_time)
    return float(cindex)


def compute_ibs(times: np.ndarray, y_train, y_test, surv_pred: np.ndarray) -> float:
    """Calculate Integrated Brier Score for survival function predictions.

    Computes the integrated Brier score (IBS) which measures the average
    squared distance between predicted survival probabilities and actual outcomes
    over a range of time points. Lower IBS indicates better calibration.

    Args:
        times: Array of time points at which to evaluate predictions.
            Shape (n_times,)
        y_train: Structured array from training set with dtype=[('event', bool), ('time', float)]
            Used to estimate censoring distribution
        y_test: Structured array from test set with dtype=[('event', bool), ('time', float)]
        surv_pred: Predicted survival probabilities with shape (n_test, n_times)
            Values should be between 0 and 1, representing P(T > t)

    Returns:
        Integrated Brier score (lower is better, typically between 0 and 0.25)

    Example:
        >>> times = np.array([3, 6, 12, 18, 24])
        >>> ibs = compute_ibs(times, y_train, y_test, survival_predictions)
        >>> print(f"IBS: {ibs:.4f}")
        IBS: 0.1523
    """
    ibs = integrated_brier_score(y_train, y_test, surv_pred, times)
    return float(ibs)


def compute_time_dependent_auc(y_train, y_test, times: np.ndarray, risk_scores) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate time-dependent AUC for survival predictions.

    Computes cumulative/dynamic AUC at multiple time points to evaluate
    the model's discriminatory ability at different horizons. AUC measures
    how well the model separates patients who experience the event before
    time t from those who survive beyond time t.

    Args:
        y_train: Structured array from training set with dtype=[('event', bool), ('time', float)]
            Used to estimate censoring distribution
        y_test: Structured array from test set with dtype=[('event', bool), ('time', float)]
        times: Array of time points at which to compute AUC. Shape (n_times,)
        risk_scores: Array of shape (n_test,) with predicted risk scores.
            Higher values indicate higher risk

    Returns:
        Tuple containing:
        - aucs: Array of AUC values at each time point, shape (n_times,)
        - mean_auc: Scalar mean AUC across all time points

    Example:
        >>> times = np.array([6, 12, 18, 24])
        >>> aucs, mean_auc = compute_time_dependent_auc(y_train, y_test, times, risk_scores)
        >>> print(f"Time-dependent AUCs: {aucs}")
        >>> print(f"Mean AUC: {mean_auc:.3f}")
        Time-dependent AUCs: [0.78 0.75 0.73 0.71]
        Mean AUC: 0.743
    """
    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
    return aucs, mean_auc