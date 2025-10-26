from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Any
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from lifelines.statistics import proportional_hazard_test
from lifelines import CoxPHFitter

from .metrics import compute_cindex, compute_ibs, compute_time_dependent_auc


@dataclass
class CVConfig:
    """Configuration for cross-validation setup.

    Attributes:
        n_splits: Number of folds for cross-validation. Defaults to 5
        random_state: Random seed for reproducibility. Defaults to 42
        shuffle: Whether to shuffle data before splitting. Defaults to True
        time_horizons: Time points (in months) for metric evaluation.
            Defaults to (3.0, 6.0, 12.0, 18.0, 24.0)
    """
    n_splits: int = 5
    random_state: int = 42
    shuffle: bool = True
    time_horizons: Iterable[float] = (3.0, 6.0, 12.0, 18.0, 24.0)


def event_balanced_splitter(y_struct, cfg: CVConfig):
    """Create stratified K-fold splits balanced on event indicator.

    Generates cross-validation splits that maintain the proportion of events
    (vs censored observations) in each fold. This ensures balanced evaluation
    across folds when dealing with censored survival data.

    Args:
        y_struct: Structured array with dtype=[('event', bool), ('time', float)]
        cfg: CVConfig instance with cross-validation parameters

    Returns:
        List of (train_indices, test_indices) tuples for each fold

    Example:
        >>> cfg = CVConfig(n_splits=5, random_state=42)
        >>> splits = event_balanced_splitter(y, cfg)
        >>> for fold_idx, (train_idx, test_idx) in enumerate(splits):
        ...     print(f"Fold {fold_idx}: {len(train_idx)} train, {len(test_idx)} test")
    """
    events = y_struct["event"].astype(int)
    skf = StratifiedKFold(
        n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=cfg.random_state
    )
    return list(skf.split(np.zeros_like(events), events))


def ph_assumption_flags(X_df: pd.DataFrame, y_struct, strata: List[str] | None = None) -> pd.DataFrame:
    """Check proportional hazards assumption using Schoenfeld residuals.

    Performs Schoenfeld residuals test for each covariate to identify potential
    violations of the proportional hazards (PH) assumption. Low p-values indicate
    covariates where PH may not hold.

    Args:
        X_df: DataFrame containing feature variables
        y_struct: Structured array with dtype=[('event', bool), ('time', float)]
        strata: Optional list of column names to stratify on. Defaults to None

    Returns:
        DataFrame with columns ['schoenfeld_p'] containing p-values for each
        covariate, sorted by p-value (most problematic first)

    Example:
        >>> ph_flags = ph_assumption_flags(X, y, strata=['risk_level_coarse'])
        >>> violations = ph_flags[ph_flags['schoenfeld_p'] < 0.05]
        >>> print(f"PH violations detected for: {violations.index.tolist()}")
    """
    df = X_df.copy()
    df["time"] = y_struct["time"]
    df["event"] = y_struct["event"].astype(int)
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event", strata=strata)
    # lifelines proportional_hazard_test on fitted model
    results = proportional_hazard_test(cph, df, time_transform="rank")
    out = (
        results.summary[["p"]]
        .rename(columns={"p": "schoenfeld_p"})
        .sort_values("schoenfeld_p")
    )
    return out


def evaluate_model(
    model_name: str,
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    times: np.ndarray,
    outdir: str,
    fold_idx: int,
    X_full: pd.DataFrame = None,
    y_full: np.ndarray = None,
    train_indices: np.ndarray = None,
    test_indices: np.ndarray = None,
    strata_cols: List[str] = None,
) -> Dict[str, Any]:
    """Evaluate survival model on single cross-validation fold.

    Fits the model pipeline on training data, generates predictions on test data,
    computes survival metrics (C-index, IBS, time-dependent AUC), and saves
    fold-specific predictions to disk for later analysis. Optionally computes
    and saves stratum-level aggregate statistics.

    Args:
        model_name: Identifier for the model being evaluated
        pipeline: sklearn Pipeline containing preprocessing and model steps
        X_train: Training feature matrix (DataFrame or array)
        y_train: Training survival data with dtype=[('event', bool), ('time', float)]
        X_test: Test feature matrix (DataFrame or array)
        y_test: Test survival data with dtype=[('event', bool), ('time', float)]
        times: Array of time points for evaluation, shape (n_times,). Will be constrained
               to test set's observed time range to prevent extrapolation errors.
        outdir: Directory path to save fold predictions
        fold_idx: Index of current fold (0-indexed)
        X_full: Full feature DataFrame with categorical strata columns (optional)
        y_full: Full structured survival array (optional)
        train_indices: Indices for training samples in full dataset (optional)
        test_indices: Indices for test samples in full dataset (optional)
        strata_cols: List of categorical column names for stratum analysis (optional)

    Returns:
        Dictionary containing:
        - model: Model name
        - fold: Fold index
        - cindex: Concordance index score
        - ibs: Integrated Brier score
        - mean_auc: Mean time-dependent AUC across time points
        - train_indices: Training indices (if provided)
        - test_indices: Test indices (if provided)
        - risk_scores: Risk scores array (for downstream analysis)
        - times: Time grid used for evaluation

    Example:
        >>> results = evaluate_model(
        ...     "cox_ph", pipeline, X_train, y_train, X_test, y_test,
        ...     times=np.array([6, 12, 18, 24]), outdir="artifacts/cox_ph", fold_idx=0,
        ...     X_full=X, y_full=y, test_indices=test_idx,
        ...     strata_cols=['typeoftariff_coarse', 'risk_level_coarse']
        ... )
        >>> print(f"Fold 0 C-index: {results['cindex']:.3f}")

    Notes:
        - Time points are automatically constrained to test set's observed range
        - This prevents "all times must be within follow-up time" errors in IBS/AUC
        - If strata_cols provided, saves per-fold stratum predictions to CSV
    """
    os.makedirs(outdir, exist_ok=True)

    # Constrain time points to test set's observed range to avoid extrapolation
    test_min = float(y_test["time"].min())
    test_max = float(y_test["time"].max())
    # Add small buffer to avoid boundary issues (exclusive upper bound in metrics)
    times_safe = times[(times >= test_min + 0.1) & (times <= test_max - 0.1)]

    # If no valid time points after filtering, use default grid for this fold
    if len(times_safe) == 0:
        from survival_framework.utils import default_time_grid
        times_safe = default_time_grid(y_train, n=50, y_test=y_test)

    # Fit
    pipe = clone(pipeline)
    pipe.fit(X_train, y_train)

    # Extract fitted estimator
    est = pipe.named_steps["model"]

    # Survival predictions (n_test, n_times) using safe time points
    surv_pred = est.predict_survival_function(
        pipe.named_steps["pre"].transform(X_test), times_safe
    ) if hasattr(est, "predict_survival_function") else None

    # Risk scores for C-index & AUC
    # Convention: negative of partial hazards -> higher risk = higher score
    if hasattr(est, "predict"):
        risk_scores = -est.predict(pipe.named_steps["pre"].transform(X_test))
    elif hasattr(est, "predict_cumulative_hazard_function"):
        chf = est.predict_cumulative_hazard_function(
            pipe.named_steps["pre"].transform(X_test)
        )
        risk_scores = np.array([fn(times_safe).max() for fn in chf])
    else:
        # fallback: 1 - survival at max time
        risk_scores = 1.0 - surv_pred[:, -1]

    # Metrics using safe time points constrained to test set range
    cindex = compute_cindex(y_train, y_test, risk_scores)
    ibs = (
        compute_ibs(times_safe, y_train, y_test, surv_pred)
        if surv_pred is not None
        else np.nan
    )

    aucs, mean_auc = compute_time_dependent_auc(y_train, y_test, times_safe, risk_scores)

    # Persist per-fold predictions for downstream ensembling/calibration
    np.save(os.path.join(outdir, f"{model_name}_fold{fold_idx}_surv.npy"), surv_pred)
    np.save(os.path.join(outdir, f"{model_name}_fold{fold_idx}_risk.npy"), risk_scores)

    # Compute and save stratum-level predictions if requested
    if strata_cols is not None and X_full is not None and test_indices is not None:
        from survival_framework.strata_analysis import compute_strata_predictions

        strata_preds = compute_strata_predictions(
            X=X_full,
            y_struct=y_full,
            strata_cols=strata_cols,
            test_indices=test_indices,
            risk_scores=risk_scores,
            surv_probs=surv_pred,
            times=times_safe,
            model_name=model_name,
            fold_idx=fold_idx
        )
        strata_path = os.path.join(outdir, f"{model_name}_fold{fold_idx}_strata.csv")
        strata_preds.to_csv(strata_path, index=False)

    # Return extended result with indices for downstream stratum analysis
    result = {
        "model": model_name,
        "fold": fold_idx,
        "cindex": cindex,
        "ibs": ibs,
        "mean_auc": float(np.mean(aucs)),
    }

    # Add optional fields for stratum metrics computation
    if train_indices is not None:
        result["train_indices"] = train_indices
    if test_indices is not None:
        result["test_indices"] = test_indices
    result["risk_scores"] = risk_scores
    result["times"] = times_safe

    return result