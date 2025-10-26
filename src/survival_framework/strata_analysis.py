"""Stratum-level aggregate statistics for survival analysis.

This module provides functions to compute and analyze predictions at the
level of categorical strata (e.g., tariff type × risk level). It enables
identification of model performance heterogeneity and segment-specific
risk profiles.

Functions:
    create_stratum_identifier: Create composite stratum labels from categorical columns
    compute_strata_summary: Compute descriptive statistics by stratum
    compute_strata_predictions: Aggregate predictions by stratum for single fold
    compute_strata_metrics: Compute performance metrics by stratum across all folds
    save_strata_artifacts: Save all stratum-level artifacts to disk
"""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from survival_framework.metrics import compute_cindex, compute_ibs, compute_time_dependent_auc


def create_stratum_identifier(
    df: pd.DataFrame,
    strata_cols: List[str]
) -> pd.Series:
    """Create composite stratum identifier from categorical columns.

    Combines multiple categorical columns into a single stratum label
    for grouping and analysis. Uses pipe separator for readability.

    Args:
        df: DataFrame containing categorical columns
        strata_cols: List of column names to combine (e.g., ['typeoftariff_coarse', 'risk_level_coarse'])

    Returns:
        Series with stratum identifiers like "tariff_A|risk_low"

    Example:
        >>> df = pd.DataFrame({
        ...     'typeoftariff_coarse': ['A', 'A', 'B'],
        ...     'risk_level_coarse': ['low', 'high', 'low']
        ... })
        >>> strata = create_stratum_identifier(df, ['typeoftariff_coarse', 'risk_level_coarse'])
        >>> print(strata.tolist())
        ['A|low', 'A|high', 'B|low']
    """
    # Combine columns with pipe separator
    stratum_parts = [df[col].astype(str) for col in strata_cols]
    return pd.Series(
        ['|'.join(parts) for parts in zip(*stratum_parts)],
        index=df.index,
        name='stratum'
    )


def compute_strata_summary(
    X: pd.DataFrame,
    y_struct: np.ndarray,
    strata_cols: List[str]
) -> pd.DataFrame:
    """Compute descriptive statistics by stratum.

    Calculates sample counts, event rates, and survival time statistics
    for each unique combination of strata variables. Useful for understanding
    data composition and baseline risk across segments.

    Args:
        X: DataFrame with features including categorical strata columns
        y_struct: Structured array with dtype=[('event', bool), ('time', float)]
        strata_cols: List of categorical column names defining strata

    Returns:
        DataFrame with columns:
        - stratum: Composite stratum identifier
        - Individual strata column values (e.g., typeoftariff_coarse, risk_level_coarse)
        - n_samples: Number of observations in stratum
        - n_events: Number of events in stratum
        - event_rate: Proportion of samples with events
        - mean_survival_time: Average survival time in months
        - median_survival_time: Median survival time in months
        - min_survival_time: Minimum survival time in months
        - max_survival_time: Maximum survival time in months

    Example:
        >>> summary = compute_strata_summary(X, y, ['typeoftariff_coarse', 'risk_level_coarse'])
        >>> print(summary[['stratum', 'n_samples', 'event_rate']].head())
    """
    # Create working DataFrame
    df = X[strata_cols].copy()
    df['event'] = y_struct['event']
    df['time'] = y_struct['time']
    df['stratum'] = create_stratum_identifier(df, strata_cols)

    # Group by stratum and compute statistics
    summary = df.groupby('stratum').agg(
        n_samples=('event', 'count'),
        n_events=('event', 'sum'),
        mean_survival_time=('time', 'mean'),
        median_survival_time=('time', 'median'),
        min_survival_time=('time', 'min'),
        max_survival_time=('time', 'max')
    ).reset_index()

    # Add event rate
    summary['event_rate'] = summary['n_events'] / summary['n_samples']

    # Add individual strata column values for easier filtering
    stratum_splits = summary['stratum'].str.split('|', expand=True)
    for i, col in enumerate(strata_cols):
        summary[col] = stratum_splits[i]

    # Reorder columns: stratum, individual cols, then stats
    cols_order = ['stratum'] + strata_cols + [
        'n_samples', 'n_events', 'event_rate',
        'mean_survival_time', 'median_survival_time',
        'min_survival_time', 'max_survival_time'
    ]
    summary = summary[cols_order]

    # Sort by sample size descending
    summary = summary.sort_values('n_samples', ascending=False).reset_index(drop=True)

    return summary


def compute_strata_predictions(
    X: pd.DataFrame,
    y_struct: np.ndarray,
    strata_cols: List[str],
    test_indices: np.ndarray,
    risk_scores: np.ndarray,
    surv_probs: Optional[np.ndarray],
    times: np.ndarray,
    model_name: str,
    fold_idx: int
) -> pd.DataFrame:
    """Aggregate predictions by stratum for a single cross-validation fold.

    Computes mean risk scores and survival probabilities at specified time
    points for each stratum. Enables analysis of model predictions across
    customer segments.

    Args:
        X: Full feature DataFrame with categorical strata columns
        y_struct: Full structured array with survival data
        strata_cols: List of categorical column names defining strata
        test_indices: Indices of test samples for this fold
        risk_scores: Array with shape (n_test,) containing predicted risk scores
        surv_probs: Array with shape (n_test, n_times) containing survival probabilities,
            or None if not available
        times: Array of time points corresponding to surv_probs columns
        model_name: Model identifier (e.g., 'cox_ph', 'gbsa')
        fold_idx: Cross-validation fold index

    Returns:
        DataFrame with columns:
        - model: Model name
        - fold: Fold index
        - stratum: Composite stratum identifier
        - n_samples: Number of test samples in stratum
        - mean_risk: Mean risk score for stratum
        - std_risk: Standard deviation of risk scores
        - mean_surv_{t}mo: Mean survival probability at t months (for each time in times)
        - std_surv_{t}mo: Std dev of survival probability at t months

    Example:
        >>> strata_preds = compute_strata_predictions(
        ...     X, y, ['typeoftariff_coarse', 'risk_level_coarse'],
        ...     test_idx, risk_scores, surv_probs, times=[6, 12, 24],
        ...     model_name='cox_ph', fold_idx=0
        ... )
    """
    # Extract test data
    X_test = X.iloc[test_indices].copy()
    y_test = y_struct[test_indices]

    # Create stratum identifier
    X_test['stratum'] = create_stratum_identifier(X_test, strata_cols)
    X_test['risk'] = risk_scores
    X_test['event'] = y_test['event']
    X_test['time'] = y_test['time']

    # Add survival probabilities for each time point
    if surv_probs is not None:
        for i, t in enumerate(times):
            X_test[f'surv_{int(t)}mo'] = surv_probs[:, i]

    # Group by stratum and compute statistics
    agg_dict = {
        'risk': ['count', 'mean', 'std']
    }

    # Add survival probability aggregations
    if surv_probs is not None:
        for t in times:
            col = f'surv_{int(t)}mo'
            agg_dict[col] = ['mean', 'std']

    grouped = X_test.groupby('stratum').agg(agg_dict)

    # Flatten multi-level columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Rename count column
    grouped = grouped.rename(columns={'risk_count': 'n_samples'})

    # Add model and fold identifiers
    grouped.insert(0, 'model', model_name)
    grouped.insert(1, 'fold', fold_idx)

    return grouped


def compute_strata_metrics(
    X: pd.DataFrame,
    y_struct: np.ndarray,
    strata_cols: List[str],
    fold_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Compute performance metrics by stratum across all cross-validation folds.

    Calculates C-index, IBS, and time-dependent AUC for each stratum in each fold.
    Enables identification of performance heterogeneity across customer segments.

    Args:
        X: Full feature DataFrame with categorical strata columns
        y_struct: Full structured array with survival data
        strata_cols: List of categorical column names defining strata
        fold_results: List of dictionaries from evaluate_model(), each containing:
            - 'test_indices': Indices of test samples
            - 'train_indices': Indices of training samples
            - 'risk_scores': Predicted risk scores
            - 'times': Time grid used for evaluation
            - 'model': Model name
            - 'fold': Fold index

    Returns:
        DataFrame with columns:
        - model: Model name
        - fold: Fold index
        - stratum: Composite stratum identifier
        - n_samples: Number of test samples in stratum
        - n_events: Number of events in stratum
        - cindex: Concordance index for stratum
        - ibs: Integrated Brier Score for stratum
        - mean_auc: Mean time-dependent AUC for stratum

    Example:
        >>> strata_metrics = compute_strata_metrics(X, y, ['typeoftariff_coarse'], fold_results)
        >>> print(strata_metrics.groupby('stratum')[['cindex', 'ibs']].mean())
    """
    results = []

    for fold_result in fold_results:
        model_name = fold_result['model']
        fold_idx = fold_result['fold']
        train_idx = fold_result['train_indices']
        test_idx = fold_result['test_indices']
        risk_scores = fold_result['risk_scores']
        times = fold_result['times']

        # Extract train and test data
        X_test = X.iloc[test_idx].copy()
        y_train = y_struct[train_idx]
        y_test = y_struct[test_idx]

        # Create stratum identifier for test set
        X_test['stratum'] = create_stratum_identifier(X_test, strata_cols)
        X_test['risk'] = risk_scores
        X_test['event'] = y_test['event']
        X_test['time'] = y_test['time']

        # Compute metrics for each stratum
        for stratum in X_test['stratum'].unique():
            stratum_mask = X_test['stratum'] == stratum
            stratum_indices = np.where(stratum_mask)[0]

            # Extract stratum data
            y_test_stratum = y_test[stratum_indices]
            risk_stratum = risk_scores[stratum_indices]

            # Skip if too few samples
            if len(y_test_stratum) < 2:
                continue

            # Compute metrics
            try:
                cindex = compute_cindex(y_train, y_test_stratum, risk_stratum)
            except Exception:
                cindex = np.nan

            # IBS requires survival probabilities (not available here, set to NaN)
            # This could be enhanced by passing surv_probs to this function
            ibs = np.nan

            try:
                aucs, mean_auc = compute_time_dependent_auc(
                    y_train, y_test_stratum, times, risk_stratum
                )
            except Exception:
                mean_auc = np.nan

            results.append({
                'model': model_name,
                'fold': fold_idx,
                'stratum': stratum,
                'n_samples': len(y_test_stratum),
                'n_events': int(y_test_stratum['event'].sum()),
                'cindex': cindex,
                'ibs': ibs,
                'mean_auc': mean_auc
            })

    return pd.DataFrame(results)


def save_strata_artifacts(
    X: pd.DataFrame,
    y_struct: np.ndarray,
    strata_cols: List[str],
    artifacts_path: str
) -> Dict[str, str]:
    """Save stratum-level summary statistics to artifacts directory.

    Computes and saves descriptive statistics for each stratum combination.
    Creates strata_summary.csv with data composition and baseline statistics.

    Args:
        X: Feature DataFrame with categorical strata columns
        y_struct: Structured survival array
        strata_cols: List of categorical column names defining strata
        artifacts_path: Path to artifacts directory

    Returns:
        Dictionary mapping artifact names to file paths

    Side Effects:
        Creates {artifacts_path}/strata_summary.csv

    Example:
        >>> paths = save_strata_artifacts(X, y, ['typeoftariff_coarse'], 'artifacts/')
        >>> print(paths['strata_summary'])
        artifacts/strata_summary.csv
    """
    os.makedirs(artifacts_path, exist_ok=True)

    # Compute summary statistics
    summary = compute_strata_summary(X, y_struct, strata_cols)

    # Save to CSV
    summary_path = os.path.join(artifacts_path, 'strata_summary.csv')
    summary.to_csv(summary_path, index=False)

    return {
        'strata_summary': summary_path
    }


def aggregate_strata_predictions(
    model_artifacts_path: str,
    model_name: str,
    n_folds: int
) -> pd.DataFrame:
    """Aggregate per-fold stratum predictions into summary statistics.

    Reads individual fold prediction files and computes mean and std
    across all folds for each stratum.

    Args:
        model_artifacts_path: Path to model's artifact directory
        model_name: Model identifier
        n_folds: Number of cross-validation folds

    Returns:
        DataFrame with mean and std of predictions across folds for each stratum

    Example:
        >>> summary = aggregate_strata_predictions('artifacts/cox_ph', 'cox_ph', n_folds=5)
        >>> print(summary[['stratum', 'mean_risk_mean', 'mean_surv_12mo_mean']])
    """
    fold_dfs = []

    # Load all fold files
    for fold_idx in range(n_folds):
        fold_file = os.path.join(
            model_artifacts_path,
            f'{model_name}_fold{fold_idx}_strata.csv'
        )
        if os.path.exists(fold_file):
            fold_dfs.append(pd.read_csv(fold_file))

    if not fold_dfs:
        return pd.DataFrame()

    # Combine all folds
    all_folds = pd.concat(fold_dfs, ignore_index=True)

    # Aggregate by stratum (mean and std across folds)
    numeric_cols = all_folds.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['fold', 'n_samples']]

    grouped = all_folds.groupby('stratum')[numeric_cols].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    grouped = grouped.reset_index()

    # Add total samples across folds
    sample_counts = all_folds.groupby('stratum')['n_samples'].sum().reset_index()
    sample_counts = sample_counts.rename(columns={'n_samples': 'total_samples'})
    grouped = grouped.merge(sample_counts, on='stratum')

    return grouped
