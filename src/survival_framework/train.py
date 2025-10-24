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
    NUM_COLS,
    load_data,
    RunType,
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
from survival_framework.utils import (
    ensure_dir,
    default_time_grid,
    save_model_metrics,
    versioned_name,
    get_output_paths,
)
from survival_framework.tracking import start_run, log_params, log_metrics, log_artifact


def _load_data_legacy(csv_path: str) -> pd.DataFrame:
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


def train_all_models(file_path: str, run_type: RunType = "sample"):
    """Train and evaluate all survival models with cross-validation.

    Complete training pipeline that:
    1. Loads data from CSV or pickle file
    2. Checks proportional hazards assumptions
    3. Trains all models with 5-fold cross-validation
    4. Computes survival metrics (C-index, IBS, time-dependent AUC)
    5. Saves per-fold predictions and final models
    6. Logs all results to MLflow
    7. Generates model ranking summary

    Args:
        file_path: Path to input file (CSV or pickle) containing survival data with
            required columns (NUM_COLS, CAT_COLS, ID_COL, TIME_COL, EVENT_COL)
        run_type: Type of run - "sample" for development, "production" for full data.
            Determines output directory structure and file naming.

    Side Effects:
        - Creates data/outputs/{run_type}/artifacts/ with PH flags, predictions, metrics
        - Creates data/outputs/{run_type}/models/ with versioned joblib files
        - Logs experiment to MLflow under "survival_framework" experiment
        - Prints progress messages and final summary paths

    Example:
        >>> # Sample run with CSV
        >>> train_all_models("data/inputs/sample/data.csv", run_type="sample")
        Loading CSV data from data/inputs/sample/data.csv (run_type=sample)
        === Training cox_ph ===
        ...

        >>> # Production run with pickle
        >>> train_all_models("data/inputs/production/data.pkl", run_type="production")
        Loading pickle data from data/inputs/production/data.pkl (run_type=production)
        ...
    """
    # Get output paths for this run type
    paths = get_output_paths(run_type)

    # Load data (supports CSV and pickle)
    df = load_data(file_path, run_type=run_type)
    X, y, ids = split_X_y(df)

    # PH check on raw features (+ categories) to flag issues
    ph_flags = ph_assumption_flags(X, y, strata=list(CAT_COLS))
    ph_flags_path = os.path.join(paths["artifacts"], "ph_flags.csv")
    ph_flags.to_csv(ph_flags_path)

    times = default_time_grid(y)

    # Detect which numeric columns are present in this dataset
    num_cols_present = [col for col in NUM_COLS if col in X.columns]
    pre = make_preprocessor(numeric=num_cols_present, categorical=CAT_COLS)

    models = build_models()

    cv_cfg = CVConfig(n_splits=5, time_horizons=(3, 6, 12, 18, 24))
    splits = event_balanced_splitter(y, cv_cfg)

    rows = []

    # MLflow run name includes run type
    run_name = f"survival_framework_{run_type}"

    with start_run(run_name=run_name):
        # Log run type as parameter
        log_params({"run_type": run_type, "input_file": file_path, "n_samples": len(df)})
        log_artifact(ph_flags_path)

        for name, model in models.items():
            print(f"\n=== Training {name} ({run_type}) ===")
            pipe = make_pipeline(pre, model)

            fold_dir = os.path.join(paths["artifacts"], f"{name}")
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
            model_filename = versioned_name(f"{name}.joblib", run_type=run_type)
            model_path = os.path.join(paths["models"], model_filename)
            joblib.dump(pipe, model_path)
            log_artifact(model_path)

    # Aggregate metrics
    metrics_df = pd.DataFrame(rows)
    metrics_path = save_model_metrics(metrics_df, paths["artifacts"])

    # Rank models
    summary = (
        metrics_df.groupby("model")[["cindex", "ibs"]].mean().reset_index()
        .assign(rank_cindex=lambda d: d["cindex"].rank(ascending=False, method="min"))
        .assign(rank_ibs=lambda d: d["ibs"].rank(ascending=True, method="min"))
        .sort_values(["rank_cindex", "rank_ibs"])
    )
    summary_path = os.path.join(paths["artifacts"], "model_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n[{run_type.upper()}] Saved:", metrics_path, summary_path)


