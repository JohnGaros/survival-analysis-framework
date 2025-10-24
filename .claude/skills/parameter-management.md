# Parameter Management Skill

Guide for documenting and organizing all configurable parameters in a centralized, maintainable way.

## Purpose

This skill provides best practices for managing different types of parameters that may vary across runs, environments, or experiments. It ensures parameters are:
- Discoverable (easy to find)
- Documented (purpose and valid ranges)
- Validated (type checking and range validation)
- Versioned (tracked in configuration files)
- Overridable (via CLI, environment, or config files)

## Parameter Categories

### 1. Model Hyperparameters

**Definition:** Parameters that control model training behavior and complexity.

**Examples:**
- `n_estimators` - Number of trees/iterations
- `learning_rate` - Step size for gradient descent
- `max_depth` - Maximum tree depth
- `l1_ratio` - L1/L2 regularization balance
- `alpha` - Regularization strength

**Best Practice: Dataclass with Environment-Specific Defaults**

```python
# src/survival_framework/config.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelHyperparameters:
    """Hyperparameters for all survival models.

    Attributes control model complexity and training behavior. Different
    values may be appropriate for sample vs production data.
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

    # Random Survival Forest
    rsf_n_estimators: int = 300
    """Number of trees in Random Survival Forest.

    Valid range: [10, 1000]
    - Sample data: 300 (high accuracy)
    - Production data: 100 (faster, still robust)

    Note: RSF benefits from more trees, but runtime is O(n_estimators).
    """

    rsf_max_depth: Optional[int] = None
    """Maximum depth of trees in RSF. None = unlimited.

    Valid range: [1, 50] or None
    - None: Grow until leaves are pure (default, can be slow)
    - 10: Good balance for large datasets
    """

    rsf_min_samples_split: int = 10
    """Minimum samples required to split a node in RSF.

    Valid range: [2, 1000]
    - Small data: 10 (default)
    - Large data: 100+ (prevents overfitting, speeds up training)
    """

    rsf_min_samples_leaf: int = 5
    """Minimum samples required in leaf node for RSF.

    Valid range: [1, 500]
    - Smaller values = more complex trees, longer training
    - Larger values = simpler trees, faster training
    """

    rsf_max_features: Optional[int] = None
    """Number of features to consider per split in RSF. None = sqrt(n_features).

    Valid range: [1, n_features] or None
    - None: Use sqrt(n_features) (default, good for most cases)
    - 5: Fixed number (useful when n_features is large)
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
                gbsa_min_samples_split=100,
                # Faster RSF for large data
                rsf_n_estimators=100,
                rsf_max_depth=10,
                rsf_min_samples_split=100,
                rsf_min_samples_leaf=50,
                rsf_max_features=5
            )
        elif run_type == "experiment":
            return cls(
                # More thorough for experiments
                gbsa_n_estimators=200,
                gbsa_learning_rate=0.05,
                rsf_n_estimators=500
            )
        else:  # sample or default
            return cls()  # Use defaults
```

### 2. Data Configuration

**Definition:** Parameters that define data sources, features, and preprocessing.

**Best Practice: Separate Config Class**

```python
# src/survival_framework/config.py

@dataclass
class DataConfig:
    """Configuration for data loading and feature selection.

    Defines which features to use, how to handle missing data, and
    preprocessing options.
    """

    # Feature columns
    numeric_features: tuple[str, ...] = (
        "debit_exp_smooth",
        "credit_exp_smooth",
        "balance_exp_smooth",
        "past_due_balance_exp_smooth",
        "oldest_past_due_exp_smooth",
        "waobd_exp_smooth",
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
```

### 3. Analysis Configuration

**Definition:** Parameters that control the analysis workflow and validation.

**Best Practice: Analysis Config Class**

```python
# src/survival_framework/config.py

@dataclass
class AnalysisConfig:
    """Configuration for survival analysis workflow.

    Controls cross-validation, time horizons for prediction, and
    evaluation metrics.
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
```

### 4. Execution Configuration

**Already Implemented:** See `src/survival_framework/config.py`

```python
@dataclass
class ExecutionConfig:
    """Configuration for execution mode and parallelization."""
    mode: ExecutionMode = ExecutionMode.PANDAS
    n_jobs: int = 1
    verbose: int = 10
    backend: str = "loky"
    # ... etc
```

## Centralized Configuration Pattern

### Main Config Class

```python
# src/survival_framework/config.py

@dataclass
class SurvivalFrameworkConfig:
    """Master configuration for survival analysis framework.

    Centralizes all configurable parameters in one place. Can be
    serialized to/from JSON or YAML for experiment tracking.

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
        """
        return cls(
            hyperparameters=ModelHyperparameters.for_environment(run_type),
            execution=ExecutionConfig.for_data_size(
                run_type == "production"
            ),
            run_type=run_type
        )

    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SurvivalFrameworkConfig":
        """Load configuration from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)

        return cls(
            hyperparameters=ModelHyperparameters(**data['hyperparameters']),
            data=DataConfig(**data['data']),
            analysis=AnalysisConfig(**data['analysis']),
            execution=ExecutionConfig(**data['execution']),
            run_type=data['run_type'],
            description=data.get('description', '')
        )
```

## Usage in Code

### When Building Models

```python
# src/survival_framework/train.py

def build_models(config: SurvivalFrameworkConfig):
    """Build models using centralized configuration."""
    hp = config.hyperparameters

    return {
        "cox_ph": CoxPHWrapper(max_iter=hp.cox_max_iter),
        "coxnet": CoxnetWrapper(l1_ratio=hp.coxnet_l1_ratio),
        "gbsa": GBSAWrapper(
            n_estimators=hp.gbsa_n_estimators,
            learning_rate=hp.gbsa_learning_rate,
            max_depth=hp.gbsa_max_depth,
            subsample=hp.gbsa_subsample,
            min_samples_split=hp.gbsa_min_samples_split
        ),
        "rsf": RSFWrapper(
            n_estimators=hp.rsf_n_estimators,
            max_depth=hp.rsf_max_depth,
            min_samples_split=hp.rsf_min_samples_split,
            min_samples_leaf=hp.rsf_min_samples_leaf,
            max_features=hp.rsf_max_features
        ),
    }
```

### CLI Integration

```python
# src/main.py

parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration JSON file"
)

if args.config:
    config = SurvivalFrameworkConfig.load(args.config)
else:
    config = SurvivalFrameworkConfig.for_run_type(args.run_type)
```

## Configuration File Examples

### configs/sample.json
```json
{
  "run_type": "sample",
  "description": "Fast configuration for testing with sample data",
  "hyperparameters": {
    "gbsa_n_estimators": 100,
    "rsf_n_estimators": 300
  },
  "execution": {
    "mode": "pandas",
    "n_jobs": 1
  }
}
```

### configs/production.json
```json
{
  "run_type": "production",
  "description": "Optimized for large datasets",
  "hyperparameters": {
    "gbsa_n_estimators": 50,
    "gbsa_subsample": 0.5,
    "rsf_n_estimators": 100,
    "rsf_max_depth": 10
  },
  "execution": {
    "mode": "mp",
    "n_jobs": 8
  }
}
```

## When to Add New Parameters

**Add a parameter when:**
1. ✅ A value appears in 2+ locations (DRY principle)
2. ✅ A value may differ across environments (sample/production/experiment)
3. ✅ A value controls important behavior (not just a constant)
4. ✅ Users might reasonably want to tune the value

**Keep as a local constant when:**
1. ✅ Value is truly constant (e.g., dtype, column order)
2. ✅ Value only used once and unlikely to change
3. ✅ Value is implementation detail (e.g., buffer sizes)

## Documentation Requirements

**For each parameter, document:**
1. **Purpose**: What does this control?
2. **Valid range**: What values are acceptable?
3. **Default**: What is the standard value?
4. **Trade-offs**: How does changing this affect behavior?
5. **Environment recommendations**: Different values for sample/production?

## Benefits of This Approach

✅ **Discoverability**: All parameters in one place
✅ **Type safety**: Dataclasses provide type checking
✅ **Validation**: Can add `__post_init__` validation
✅ **Documentation**: Docstrings explain each parameter
✅ **Versioning**: Config files can be tracked in git
✅ **Reproducibility**: Save/load exact configurations
✅ **Testing**: Easy to create test configurations
✅ **Experimentation**: Create variants for A/B testing
