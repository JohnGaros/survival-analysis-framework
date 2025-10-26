"""Configuration module for execution modes and parallelization settings.

This module provides configuration for different execution modes:
- pandas: Single-threaded pandas (default, backward compatible)
- multiprocessing: Parallel cross-validation using joblib
- pyspark_local: PySpark in local mode (future)
- pyspark_cluster: Distributed PySpark cluster (future)
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import os
import multiprocessing
import json


class ExecutionMode(str, Enum):
    """Execution mode for survival analysis pipeline.

    Attributes:
        PANDAS: Single-threaded pandas execution (default, backward compatible)
        MULTIPROCESSING: Parallel cross-validation using multiprocessing
        PYSPARK_LOCAL: PySpark in local mode (future implementation)
        PYSPARK_CLUSTER: Distributed PySpark cluster (future implementation)
    """
    PANDAS = "pandas"
    MULTIPROCESSING = "mp"
    PYSPARK_LOCAL = "spark_local"
    PYSPARK_CLUSTER = "spark_cluster"


@dataclass
class ExecutionConfig:
    """Configuration for execution mode and parallelization.

    Attributes:
        mode: Execution mode (pandas, mp, spark_local, spark_cluster)
        n_jobs: Number of parallel jobs. -1 means use all cores, 1 means sequential
        verbose: Verbosity level for joblib (0=silent, 10=progress bar, 50=detailed)
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')
        auto_mode: If True, automatically select mode based on data size
        size_threshold_mb: Data size threshold for auto-selecting execution mode
        spark_master: Spark master URL (for spark modes)
        spark_memory: Memory allocation for Spark executor

    Example:
        >>> # Default configuration (pandas, single-threaded)
        >>> config = ExecutionConfig()

        >>> # Multiprocessing with all cores
        >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=-1)

        >>> # Auto-detect mode based on data size
        >>> config = ExecutionConfig(auto_mode=True)
    """
    mode: ExecutionMode = ExecutionMode.PANDAS
    n_jobs: int = 1
    verbose: int = 10
    backend: str = "loky"
    auto_mode: bool = False
    size_threshold_mb: float = 10.0
    spark_master: str = "local[*]"
    spark_memory: str = "4g"

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Convert string mode to enum if needed
        if isinstance(self.mode, str):
            self.mode = ExecutionMode(self.mode)

        # Normalize n_jobs
        if self.n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif self.n_jobs < 1:
            raise ValueError(f"n_jobs must be -1 or positive, got {self.n_jobs}")

        # For pandas mode, force n_jobs=1
        if self.mode == ExecutionMode.PANDAS:
            self.n_jobs = 1

    def get_effective_n_jobs(self) -> int:
        """Get the effective number of jobs for parallel execution.

        Returns:
            Number of parallel jobs to use (1 for sequential, >1 for parallel)

        Example:
            >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=-1)
            >>> config.get_effective_n_jobs()
            8  # On an 8-core machine
        """
        if self.mode == ExecutionMode.PANDAS:
            return 1
        return self.n_jobs

    def is_parallel(self) -> bool:
        """Check if parallel execution is enabled.

        Returns:
            True if execution mode supports parallelism and n_jobs > 1

        Example:
            >>> config = ExecutionConfig(mode=ExecutionMode.PANDAS)
            >>> config.is_parallel()
            False

            >>> config = ExecutionConfig(mode=ExecutionMode.MULTIPROCESSING, n_jobs=4)
            >>> config.is_parallel()
            True
        """
        return self.mode != ExecutionMode.PANDAS and self.n_jobs > 1

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"ExecutionConfig(mode={self.mode.value}, "
            f"n_jobs={self.n_jobs}, "
            f"parallel={self.is_parallel()})"
        )


def select_execution_mode(
    data_size_mb: float,
    n_cores: Optional[int] = None,
    force_mode: Optional[ExecutionMode] = None
) -> ExecutionMode:
    """Auto-select execution mode based on data characteristics.

    Selection logic:
    - Small data (<10MB): pandas (overhead not worth it)
    - Medium data (10-100MB): multiprocessing (parallel CV)
    - Large data (>100MB): multiprocessing (or spark if available)

    Args:
        data_size_mb: Dataset size in megabytes
        n_cores: Number of available CPU cores (auto-detected if None)
        force_mode: Force a specific mode (overrides auto-detection)

    Returns:
        Recommended execution mode

    Example:
        >>> # Small data: use pandas
        >>> select_execution_mode(5.0)
        <ExecutionMode.PANDAS: 'pandas'>

        >>> # Medium data: use multiprocessing
        >>> select_execution_mode(50.0)
        <ExecutionMode.MULTIPROCESSING: 'mp'>

        >>> # Force a specific mode
        >>> select_execution_mode(50.0, force_mode=ExecutionMode.PANDAS)
        <ExecutionMode.PANDAS: 'pandas'>
    """
    if force_mode is not None:
        return force_mode

    if n_cores is None:
        n_cores = multiprocessing.cpu_count()

    # Small data: pandas (overhead not worth it)
    if data_size_mb < 10.0:
        return ExecutionMode.PANDAS

    # Medium to large data: use multiprocessing if we have multiple cores
    if n_cores > 1:
        return ExecutionMode.MULTIPROCESSING

    # Fallback to pandas for single-core machines
    return ExecutionMode.PANDAS


def create_execution_config(
    mode: Optional[str] = None,
    n_jobs: int = -1,
    auto_mode: bool = True,
    verbose: int = 10
) -> ExecutionConfig:
    """Factory function to create ExecutionConfig with sensible defaults.

    Args:
        mode: Execution mode string ('pandas', 'mp', 'spark_local', 'spark_cluster')
        n_jobs: Number of parallel jobs (-1 = all cores)
        auto_mode: Enable auto-detection of execution mode
        verbose: Verbosity level (0=silent, 10=progress, 50=detailed)

    Returns:
        ExecutionConfig instance

    Example:
        >>> # Default: auto-detect mode, use all cores
        >>> config = create_execution_config()

        >>> # Force multiprocessing with 4 jobs
        >>> config = create_execution_config(mode='mp', n_jobs=4)

        >>> # Silent execution
        >>> config = create_execution_config(verbose=0)
    """
    if mode is None:
        execution_mode = ExecutionMode.PANDAS if not auto_mode else ExecutionMode.MULTIPROCESSING
    else:
        execution_mode = ExecutionMode(mode)

    return ExecutionConfig(
        mode=execution_mode,
        n_jobs=n_jobs,
        auto_mode=auto_mode,
        verbose=verbose
    )


def get_data_size_mb(file_path: str) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to data file

    Returns:
        File size in MB

    Example:
        >>> get_data_size_mb('data/inputs/production/large_data.pkl')
        30.5
    """
    if not os.path.exists(file_path):
        return 0.0

    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


# ============================================================================
# Model Hyperparameters Configuration
# ============================================================================

@dataclass
class ModelHyperparameters:
    """Hyperparameters for all survival models.

    Attributes control model complexity and training behavior. Different
    values may be appropriate for sample vs production data.

    Attributes:
        cox_max_iter: Maximum iterations for Cox model optimization
        coxnet_l1_ratio: Balance between L1 (1.0) and L2 (0.0) penalty for CoxNet
        coxnet_alpha_min_ratio: Ratio of smallest to largest alpha in path
        coxnet_n_alphas: Number of alphas in regularization path
        gbsa_n_estimators: Number of boosting iterations for GBSA
        gbsa_learning_rate: Learning rate (shrinkage) for GBSA
        gbsa_max_depth: Maximum depth of individual trees in GBSA
        gbsa_subsample: Fraction of samples used for each tree in GBSA
        gbsa_min_samples_split: Minimum samples required to split a node in GBSA
    """
    # Cox PH models
    cox_max_iter: int = 10_000
    """Maximum iterations for Cox model optimization."""

    coxnet_l1_ratio: float = 0.5
    """Balance between L1 (1.0) and L2 (0.0) penalty for CoxNet.

    Valid range: [0.0, 1.0]
    - 0.0 = Ridge (L2 only)
    - 1.0 = Lasso (L1 only)
    - 0.5 = Elastic net (balanced)
    """

    coxnet_alpha_min_ratio: float = 0.01
    """Ratio of smallest to largest alpha in regularization path."""

    coxnet_n_alphas: int = 100
    """Number of alphas in regularization path for cross-validation."""

    # Gradient Boosting
    gbsa_n_estimators: int = 100
    """Number of boosting iterations for GBSA.

    Valid range: [10, 1000]
    - Sample data: 100 (default)
    - Production data: 50 (faster, still accurate)

    Note: Larger values increase accuracy but runtime scales linearly.
    """

    gbsa_learning_rate: float = 0.1
    """Learning rate (shrinkage) for GBSA.

    Valid range: [0.001, 1.0]
    - Lower values require more estimators but may generalize better
    - Higher values converge faster but may overfit
    """

    gbsa_max_depth: int = 3
    """Maximum depth of individual trees in GBSA.

    Valid range: [1, 10]
    - Shallow trees (1-3): Fast, less overfitting, good for linear patterns
    - Deep trees (5-10): Slow, captures complex interactions
    """

    gbsa_subsample: float = 1.0
    """Fraction of samples used for each tree in GBSA.

    Valid range: [0.1, 1.0]
    - 1.0: Use all samples (default)
    - 0.5: Use 50% (faster, adds randomness, reduces overfitting)
    """

    gbsa_min_samples_split: int = 2
    """Minimum samples required to split a node in GBSA.

    Valid range: [2, 1000]
    - Small data: 2 (default)
    - Large data: 100+ (prevents overfitting, speeds up training)
    """

    @classmethod
    def for_environment(cls, run_type: str) -> "ModelHyperparameters":
        """Create hyperparameters optimized for specific environment.

        Args:
            run_type: One of "sample", "production", "experiment"

        Returns:
            ModelHyperparameters instance with appropriate defaults

        Example:
            >>> params = ModelHyperparameters.for_environment("production")
            >>> params.gbsa_n_estimators
            50
        """
        if run_type == "production":
            return cls(
                # Faster GBSA for large data
                gbsa_n_estimators=50,
                gbsa_learning_rate=0.2,
                gbsa_max_depth=2,
                gbsa_subsample=0.5,
                gbsa_min_samples_split=100
            )
        elif run_type == "experiment":
            return cls(
                # More thorough for experiments
                gbsa_n_estimators=200,
                gbsa_learning_rate=0.05
            )
        else:  # sample or default
            return cls()  # Use defaults


# ============================================================================
# Data Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and feature selection.

    Defines which features to use, how to handle missing data, and
    preprocessing options.

    Attributes:
        numeric_features: Tuple of numeric feature column names
        categorical_features: Tuple of categorical feature column names
        stratification_columns: Tuple of columns for stratified Cox models
        time_column: Column containing survival time
        event_column: Column containing event indicator (True/1 = event occurred)
        id_column: Column containing unique identifiers for samples
        variance_threshold: Minimum variance required for feature inclusion
        missing_indicator: Whether to add binary indicators for missing values
        imputation_strategy: Strategy for imputing missing numeric values
        min_survival_time: Minimum valid survival time (records below are filtered)
    """
    # Feature columns
    numeric_features: tuple[str, ...] = (
        "debit_exp_smooth",
        "credit_exp_smooth",
        "balance_exp_smooth",
        "past_due_balance_exp_smooth",
        "oldest_past_due_exp_smooth",
        "waobd_exp_smooth",
        "kwh_exp_smooth",
        "total_settlements",
        "active_settlements",
        "defaulted_settlements"
    )
    """Numeric features for survival model.

    All features should be continuous or count variables that benefit
    from standardization. Add/remove features here to experiment with
    different feature sets.
    """

    categorical_features: tuple[str, ...] = (
        "typeoftariff_coarse",
        "risk_level_coarse"
    )
    """Categorical features for survival model.

    Features will be one-hot encoded during preprocessing. Ensure
    categories are properly labeled in raw data.
    """

    stratification_columns: tuple[str, ...] = (
        "typeoftariff_coarse",
        "risk_level_coarse"
    )
    """Columns to use for stratified Cox models.

    These columns define strata for handling non-proportional hazards.
    Must be subset of categorical_features.
    """

    time_column: str = "survival_months"
    """Column containing survival time."""

    event_column: str = "is_terminated"
    """Column containing event indicator (True/1 = event occurred)."""

    id_column: str = "account_entities_key"
    """Column containing unique identifiers for samples."""

    # Preprocessing options
    variance_threshold: float = 0.01
    """Minimum variance required for feature to be included.

    Valid range: [0.0, 1.0]
    Features with variance below this threshold are removed as they
    provide little discriminative information.
    """

    missing_indicator: bool = True
    """Whether to add binary indicators for missing values.

    When True, adds an indicator column for each feature with missing
    values before imputation. This preserves information about
    missingness patterns.
    """

    imputation_strategy: str = "mean"
    """Strategy for imputing missing numeric values.

    Valid options: "mean", "median", "most_frequent", "constant"
    """

    min_survival_time: float = 0.0
    """Minimum valid survival time. Records below this are filtered.

    Valid range: [0.0, inf)
    Use > 0.0 to remove zero or negative survival times which are
    typically data errors.
    """


# ============================================================================
# Analysis Configuration
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for survival analysis workflow.

    Controls cross-validation, time horizons for prediction, and
    evaluation metrics.

    Attributes:
        n_folds: Number of cross-validation folds
        cv_random_state: Random seed for reproducible CV splits
        prediction_horizons: Time points (months) for time-dependent AUC evaluation
        time_grid_points: Number of time points for survival function evaluation
        compute_brier_score: Whether to compute Integrated Brier Score (IBS)
        compute_time_dependent_auc: Whether to compute time-dependent AUC
    """
    # Cross-validation
    n_folds: int = 5
    """Number of cross-validation folds.

    Valid range: [2, 10]
    - 5: Standard, good balance of bias-variance
    - 10: More thorough but slower
    """

    cv_random_state: int = 42
    """Random seed for reproducible cross-validation splits."""

    # Time horizons
    prediction_horizons: tuple[int, ...] = (3, 6, 12, 18, 24)
    """Time points (months) for time-dependent AUC evaluation.

    These should cover the range of clinically relevant time periods.
    Modify based on your domain:
    - Short-term: 3, 6 months
    - Medium-term: 12, 18 months
    - Long-term: 24+ months
    """

    time_grid_points: int = 100
    """Number of time points for survival function evaluation.

    Valid range: [10, 1000]
    More points = smoother curves but more computation.
    """

    # Metrics
    compute_brier_score: bool = True
    """Whether to compute Integrated Brier Score (IBS).

    IBS measures calibration but can be expensive for large datasets.
    Set to False to speed up evaluation if only discrimination
    (C-index) is needed.
    """

    compute_time_dependent_auc: bool = True
    """Whether to compute time-dependent AUC at prediction horizons.

    Time-dependent AUC shows how discrimination changes over time.
    Set to False to speed up evaluation.
    """


# ============================================================================
# Master Configuration
# ============================================================================

@dataclass
class SurvivalFrameworkConfig:
    """Master configuration for survival analysis framework.

    Centralizes all configurable parameters in one place. Can be
    serialized to/from JSON for experiment tracking.

    Attributes:
        hyperparameters: Model hyperparameter configuration
        data: Data loading and preprocessing configuration
        analysis: Cross-validation and evaluation configuration
        execution: Execution mode and parallelization configuration
        run_type: Type of run ("sample", "production", "experiment")
        description: Optional description of this configuration

    Example:
        >>> config = SurvivalFrameworkConfig.for_run_type("production")
        >>> config.hyperparameters.gbsa_n_estimators
        50
        >>> config.save("configs/production.json")
        >>> loaded = SurvivalFrameworkConfig.load("configs/production.json")
    """
    # Sub-configurations
    hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Run metadata
    run_type: str = "sample"
    """Type of run: 'sample', 'production', or 'experiment'."""

    description: str = ""
    """Optional description of this configuration."""

    @classmethod
    def for_run_type(cls, run_type: str) -> "SurvivalFrameworkConfig":
        """Create configuration optimized for specific run type.

        Args:
            run_type: One of "sample", "production", "experiment"

        Returns:
            Configured instance with appropriate defaults

        Example:
            >>> config = SurvivalFrameworkConfig.for_run_type("production")
            >>> config.hyperparameters.gbsa_n_estimators
            50
        """
        # Create execution config based on run type
        if run_type == "production":
            exec_config = ExecutionConfig(
                mode=ExecutionMode.MULTIPROCESSING,
                n_jobs=-1,
                verbose=10
            )
        else:
            exec_config = ExecutionConfig()

        return cls(
            hyperparameters=ModelHyperparameters.for_environment(run_type),
            execution=exec_config,
            run_type=run_type
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization.

        Returns:
            Dictionary representation of configuration

        Example:
            >>> config = SurvivalFrameworkConfig.for_run_type("sample")
            >>> config_dict = config.to_dict()
            >>> config_dict['run_type']
            'sample'
        """
        def _dataclass_to_dict(obj):
            """Recursively convert dataclass to dict."""
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    k: _dataclass_to_dict(v)
                    for k, v in obj.__dict__.items()
                }
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        return _dataclass_to_dict(self)

    def save(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to output JSON file

        Example:
            >>> config = SurvivalFrameworkConfig.for_run_type("production")
            >>> config.save("configs/production.json")
        """
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SurvivalFrameworkConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to input JSON file

        Returns:
            SurvivalFrameworkConfig instance

        Example:
            >>> config = SurvivalFrameworkConfig.load("configs/production.json")
            >>> config.hyperparameters.gbsa_n_estimators
            50
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        return cls(
            hyperparameters=ModelHyperparameters(**data['hyperparameters']),
            data=DataConfig(**data['data']),
            analysis=AnalysisConfig(**data['analysis']),
            execution=ExecutionConfig(**data['execution']),
            run_type=data['run_type'],
            description=data.get('description', '')
        )
