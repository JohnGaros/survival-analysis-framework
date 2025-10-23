from __future__ import annotations
import os
import joblib
import pandas as pd
import numpy as np

from survival_framework.data import (
    split_X_y,
    make_preprocessor,
    make_pipeline,
    CAT_COLS,
)
from survival_framework.models import (
    CoxPHWrapper,
    CoxnetWrapper,
    StratifiedCoxWrapper,
    WeibullAFTWrapper,
    GBSAWrapper,
    RSFWrapper,
)
from survival_framework.validation import (
    CVConfig,
    event_balanced_splitter,
    evaluate_model,
    ph_assumption_flags,
)
from survival_framework.utils import ensure_dir, default_time_grid, save_model_metrics, versioned_name
from survival_framework.tracking import start_run, log_params, log_metrics, log_artifact


ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
ensure_dir(ARTIFACTS_DIR)
ensure_dir(MODELS_DIR)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load survival data from CSV file.

    Args:
        csv_path: Path to CSV file containing survival data

    Returns:
        DataFrame with survival data including features, time, and event columns

    Example:
        >>> df = load_data("data/survival_inputs.csv")
        >>> print(df.shape)
        (2000, 15)
    """
    return pd.read_csv(csv_path)


def build_models():
    """Construct dictionary of all survival models to train.

    Instantiates all model wrappers with default or specified hyperparameters.
    Models include Cox PH, regularized Cox, Weibull AFT, gradient boosting,
    and random survival forest.

    Returns:
        Dictionary mapping model names to instantiated model wrapper objects

    Example:
        >>> models = build_models()
        >>> print(list(models.keys()))
        ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']

    Note:
        StratifiedCoxWrapper is excluded because it requires DataFrame input with
        original categorical columns, which is incompatible with the preprocessing
        pipeline (specifically VarianceThreshold which converts to numpy arrays).
        To use StratifiedCoxWrapper, train it separately without the pipeline.
    """
    return {
        "cox_ph": CoxPHWrapper(),
        "coxnet": CoxnetWrapper(l1_ratio=0.5),
        # "cox_stratified": StratifiedCoxWrapper(),  # Incompatible with pipeline - requires raw DataFrame
        "weibull_aft": WeibullAFTWrapper(),
        "gbsa": GBSAWrapper(),
        "rsf": RSFWrapper(),
    }


def train_all_models(csv_path: str):
    """Train and evaluate all survival models with cross-validation.

    Complete training pipeline that:
    1. Loads data and performs train/test splits
    2. Checks proportional hazards assumptions
    3. Trains all models with 5-fold cross-validation
    4. Computes survival metrics (C-index, IBS, time-dependent AUC)
    5. Saves per-fold predictions and final models
    6. Logs all results to MLflow
    7. Generates model ranking summary

    Args:
        csv_path: Path to CSV file containing survival data with required columns
            (NUM_COLS, CAT_COLS, ID_COL, TIME_COL, EVENT_COL)

    Side Effects:
        - Creates artifacts/ directory with PH flags, per-fold predictions, and metrics
        - Creates models/ directory with versioned joblib files for each model
        - Logs experiment to MLflow under "survival_framework" experiment
        - Prints progress messages and final summary paths

    Example:
        >>> train_all_models("data/survival_inputs_sample2000.csv")
        === Training cox_ph ===
        === Training coxnet ===
        ...
        Saved: artifacts/model_metrics.csv artifacts/model_summary.csv
    """
    df = load_data(csv_path)
    X, y, ids = split_X_y(df)

    # PH check on raw features (+ categories) to flag issues
    ph_flags = ph_assumption_flags(X, y, strata=list(CAT_COLS))
    ph_flags_path = os.path.join(ARTIFACTS_DIR, "ph_flags.csv")
    ph_flags.to_csv(ph_flags_path)

    times = default_time_grid(y)

    pre = make_preprocessor()

    models = build_models()

    cv_cfg = CVConfig(n_splits=5, time_horizons=(3, 6, 12, 18, 24))
    splits = event_balanced_splitter(y, cv_cfg)

    rows = []

    with start_run(run_name="survival_framework_run"):
        log_artifact(ph_flags_path)

        for name, model in models.items():
            print(f"\n=== Training {name} ===")
            pipe = make_pipeline(pre, model)

            fold_dir = os.path.join(ARTIFACTS_DIR, f"{name}")
            ensure_dir(fold_dir)

            for fold_idx, (tr, te) in enumerate(splits):
                X_train, X_test = X.iloc[tr], X.iloc[te]
                y_train, y_test = y[tr], y[te]

                res = evaluate_model(
                    model_name=name,
                    pipeline=pipe,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    times=times,
                    outdir=fold_dir,
                    fold_idx=fold_idx,
                )
                rows.append(res)
                log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx)

            # Save fitted final model on full data
            pipe.fit(X, y)
            model_path = os.path.join(MODELS_DIR, versioned_name(f"{name}.joblib"))
            joblib.dump(pipe, model_path)
            log_artifact(model_path)

    # Aggregate metrics
    metrics_df = pd.DataFrame(rows)
    metrics_path = save_model_metrics(metrics_df, ARTIFACTS_DIR)

    # Rank models
    summary = (
        metrics_df.groupby("model")[["cindex", "ibs"]].mean().reset_index()
        .assign(rank_cindex=lambda d: d["cindex"].rank(ascending=False, method="min"))
        .assign(rank_ibs=lambda d: d["ibs"].rank(ascending=True, method="min"))
        .sort_values(["rank_cindex", "rank_ibs"])
    )
    summary_path = os.path.join(ARTIFACTS_DIR, "model_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Saved:", metrics_path, summary_path)


