from __future__ import annotations
import os
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
from typing import Optional

from survival_framework.data import (
    split_X_y,
    make_preprocessor,
    make_pipeline,
    CAT_COLS,
    NUM_COLS,
    load_data,
    RunType,
)
from survival_framework.config import ExecutionConfig, ExecutionMode
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


def build_models(hyperparameters=None):
    """Construct dictionary of all survival models to train.

    Instantiates all model wrappers with hyperparameters from config or defaults.
    Models include Cox PH, regularized Cox, Weibull AFT, gradient boosting,
    and random survival forest.

    Args:
        hyperparameters: ModelHyperparameters instance with configured parameters.
            If None, uses default hyperparameters.

    Returns:
        Dictionary mapping model names to instantiated model wrapper objects

    Example:
        >>> # Using defaults
        >>> models = build_models()
        >>> print(list(models.keys()))
        ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']

        >>> # Using custom hyperparameters
        >>> from survival_framework.config import ModelHyperparameters
        >>> hp = ModelHyperparameters.for_environment("production")
        >>> models = build_models(hp)
        >>> # GBSA will use production-optimized parameters

    Note:
        StratifiedCoxWrapper is excluded because it requires DataFrame input with
        original categorical columns, which is incompatible with the preprocessing
        pipeline (specifically VarianceThreshold which converts to numpy arrays).
        To use StratifiedCoxWrapper, train it separately without the pipeline.
    """
    if hyperparameters is None:
        from survival_framework.config import ModelHyperparameters
        hyperparameters = ModelHyperparameters()

    return {
        "cox_ph": CoxPHWrapper(
            max_iter=hyperparameters.cox_max_iter
        ),
        "coxnet": CoxnetWrapper(
            l1_ratio=hyperparameters.coxnet_l1_ratio,
            alpha_min_ratio=hyperparameters.coxnet_alpha_min_ratio,
            n_alphas=hyperparameters.coxnet_n_alphas
        ),
        # "cox_stratified": StratifiedCoxWrapper(),  # Incompatible with pipeline - requires raw DataFrame
        "weibull_aft": WeibullAFTWrapper(),
        "gbsa": GBSAWrapper(
            n_estimators=hyperparameters.gbsa_n_estimators,
            learning_rate=hyperparameters.gbsa_learning_rate,
            max_depth=hyperparameters.gbsa_max_depth,
            subsample=hyperparameters.gbsa_subsample,
            min_samples_split=hyperparameters.gbsa_min_samples_split
        ),
        "rsf": RSFWrapper(
            n_estimators=hyperparameters.rsf_n_estimators,
            max_depth=hyperparameters.rsf_max_depth,
            min_samples_split=hyperparameters.rsf_min_samples_split,
            min_samples_leaf=hyperparameters.rsf_min_samples_leaf,
            max_features=hyperparameters.rsf_max_features
        ),
    }


def _train_single_fold(
    model_name: str,
    pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    times: np.ndarray,
    fold_dir: str
) -> dict:
    """Train and evaluate a single fold (helper for parallel execution).

    Args:
        model_name: Name of the model being trained
        pipeline: sklearn Pipeline with preprocessor and model
        X: Full feature matrix
        y: Full structured survival array
        fold_idx: Fold index (0-based)
        train_idx: Indices for training set
        test_idx: Indices for test set
        times: Time grid for evaluation
        fold_dir: Directory to save fold predictions

    Returns:
        Dictionary with evaluation metrics (cindex, ibs, auc_mean, etc.)
    """
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    res = evaluate_model(
        model_name=model_name,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        times=times,
        outdir=fold_dir,
        fold_idx=fold_idx,
    )
    return res


def train_all_models(
    file_path: str,
    run_type: RunType = "sample",
    execution_config: Optional[ExecutionConfig] = None,
    config: Optional["SurvivalFrameworkConfig"] = None
):
    """Train and evaluate all survival models with cross-validation.

    Complete training pipeline that:
    1. Loads data from CSV or pickle file
    2. Checks proportional hazards assumptions
    3. Trains all models with 5-fold cross-validation (sequential or parallel)
    4. Computes survival metrics (C-index, IBS, time-dependent AUC)
    5. Saves per-fold predictions and final models
    6. Logs all results to MLflow
    7. Generates model ranking summary

    Args:
        file_path: Path to input file (CSV or pickle) containing survival data with
            required columns (NUM_COLS, CAT_COLS, ID_COL, TIME_COL, EVENT_COL)
        run_type: Type of run - "sample" for development, "production" for full data.
            Determines output directory structure and file naming.
        execution_config: Configuration for execution mode and parallelization.
            If None, defaults to pandas mode (sequential, backward compatible).
            Ignored if config is provided.
        config: Complete SurvivalFrameworkConfig with all parameters.
            If provided, takes precedence over execution_config.

    Side Effects:
        - Creates data/outputs/{run_type}/artifacts/ with PH flags, predictions, metrics
        - Creates data/outputs/{run_type}/models/ with versioned joblib files
        - Logs experiment to MLflow under "survival_framework" experiment
        - Prints progress messages and final summary paths

    Example:
        >>> # Sample run with CSV (backward compatible)
        >>> train_all_models("data/inputs/sample/data.csv", run_type="sample")
        Loading CSV data from data/inputs/sample/data.csv (run_type=sample)
        === Training cox_ph ===
        ...

        >>> # Production run with full config
        >>> from survival_framework.config import SurvivalFrameworkConfig
        >>> cfg = SurvivalFrameworkConfig.for_run_type("production")
        >>> train_all_models("data/inputs/production/data.pkl", config=cfg)
        Loading pickle data from data/inputs/production/data.pkl (run_type=production)
        ...
    """
    # Handle configuration
    if config is not None:
        # Use full config if provided
        hyperparameters = config.hyperparameters
        execution_config = config.execution
        run_type = config.run_type
    else:
        # Backward compatible: use defaults
        from survival_framework.config import ModelHyperparameters
        hyperparameters = ModelHyperparameters.for_environment(run_type)
        if execution_config is None:
            execution_config = ExecutionConfig(mode=ExecutionMode.PANDAS, n_jobs=1)

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

    models = build_models(hyperparameters)

    cv_cfg = CVConfig(n_splits=5, time_horizons=(3, 6, 12, 18, 24))
    splits = event_balanced_splitter(y, cv_cfg)

    rows = []

    # MLflow run name includes run type
    run_name = f"survival_framework_{run_type}"

    with start_run(run_name=run_name):
        # Log run type and execution config
        log_params({
            "run_type": run_type,
            "input_file": file_path,
            "n_samples": len(df),
            "execution_mode": execution_config.mode.value,
            "n_jobs": execution_config.n_jobs
        })
        log_artifact(ph_flags_path)

        for name, model in models.items():
            print(f"\n=== Training {name} ({run_type}, {execution_config.mode.value}, n_jobs={execution_config.n_jobs}) ===")
            pipe = make_pipeline(pre, model)

            fold_dir = os.path.join(paths["artifacts"], f"{name}")
            ensure_dir(fold_dir)

            # Convert splits to list for parallel execution
            splits_list = list(splits)

            # Decide execution strategy
            if execution_config.is_parallel():
                # Parallel execution using joblib
                print(f"  → Parallel CV with {execution_config.n_jobs} jobs")
                fold_results = Parallel(
                    n_jobs=execution_config.n_jobs,
                    verbose=execution_config.verbose,
                    backend=execution_config.backend
                )(
                    delayed(_train_single_fold)(
                        model_name=name,
                        pipeline=pipe,
                        X=X,
                        y=y,
                        fold_idx=fold_idx,
                        train_idx=tr,
                        test_idx=te,
                        times=times,
                        fold_dir=fold_dir
                    )
                    for fold_idx, (tr, te) in enumerate(splits_list)
                )

                # Collect results and log metrics
                for fold_idx, res in enumerate(fold_results):
                    rows.append(res)
                    log_metrics({f"{name}_cindex": res["cindex"], f"{name}_ibs": res["ibs"]}, step=fold_idx)
            else:
                # Sequential execution (original behavior)
                print(f"  → Sequential CV")
                for fold_idx, (tr, te) in enumerate(splits_list):
                    res = _train_single_fold(
                        model_name=name,
                        pipeline=pipe,
                        X=X,
                        y=y,
                        fold_idx=fold_idx,
                        train_idx=tr,
                        test_idx=te,
                        times=times,
                        fold_dir=fold_dir
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


