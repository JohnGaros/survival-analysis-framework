from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Any
import numpy as np
import pandas as pd

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.util import Surv
from lifelines import CoxPHFitter, WeibullAFTFitter


class BaseSurvivalModel:
    """Base class for survival model wrappers with unified interface.

    Provides a common API for different survival model implementations,
    ensuring consistent fit, predict, and score methods across all models.

    Attributes:
        name: String identifier for the model type
    """

    name: str = "base"

    def fit(self, X, y):
        """Fit the survival model to training data.

        Args:
            X: Feature matrix (DataFrame or array)
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival probabilities at specified time points.

        Args:
            X: Feature matrix for prediction
            times: Time points at which to evaluate survival function

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError

    def score(self, X, y) -> float:
        """Calculate concordance index for model predictions.

        Computes Harrell's C-index to evaluate model discrimination ability.

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index (C-index) score, typically between 0.5 and 1.0

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


@dataclass
class CoxPHWrapper(BaseSurvivalModel):
    """Wrapper for scikit-survival Cox Proportional Hazards model with L2 regularization.

    Implements the Cox proportional hazards model for survival analysis with
    Ridge (L2) regularization to stabilize coefficient estimates and prevent
    numerical issues from multicollinearity.

    Attributes:
        name: Model identifier, defaults to "cox_ph"
        model: Underlying CoxPHSurvivalAnalysis instance
        alpha: L2 regularization strength. Higher values increase regularization.
               Default is 1.0 which provides numerical stability while allowing
               reasonable coefficient magnitudes.

    Example:
        >>> cox = CoxPHWrapper()  # Uses alpha=1.0 by default
        >>> cox.fit(X_train, y_train)
        >>> risk_scores = cox.predict(X_test)

        >>> # Custom regularization
        >>> cox_heavy = CoxPHWrapper(alpha=10.0)  # More regularization
        >>> cox_light = CoxPHWrapper(alpha=0.1)   # Less regularization

    Notes:
        - L2 regularization (alpha > 0) prevents LAPACK errors from:
          * Multicollinearity (correlated features)
          * High-dimensional data (many features relative to samples)
          * Near-singular covariance matrices
        - Default alpha=1.0 balances stability and model flexibility
        - Set alpha=0 for unregularized Cox (not recommended with high-dimensional data)
    """
    name: str = "cox_ph"
    model: CoxPHSurvivalAnalysis = None
    alpha: float = 1.0
    max_iter: int = 10_000

    def __post_init__(self):
        """Initialize the underlying Cox PH model with regularization."""
        if self.model is None:
            self.model = CoxPHSurvivalAnalysis(
                alpha=self.alpha,  # L2 regularization for numerical stability
                n_iter=self.max_iter,
                tol=1e-9
            )

    def fit(self, X, y):
        """Fit Cox PH model with input validation.

        Validates input data for finite values, minimum sample size,
        and feature variance before fitting.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance

        Raises:
            ValueError: If X contains non-finite values, has fewer than 2 samples,
                or contains zero-variance features
        """
        X_array = np.asarray(X)
        if not np.isfinite(X_array).all():
            raise ValueError(f"{self.name}: Non-finite values detected in X before fitting")
        if X_array.shape[0] < 2:
            raise ValueError(f"{self.name}: Need at least 2 samples to fit")
        if np.any(np.std(X_array, axis=0) == 0):
            raise ValueError(f"{self.name}: Zero-variance features detected")

        self.model.fit(X, y)
        return self

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities
        """
        sfns = self.model.predict_survival_function(X)
        times = np.asarray(list(times), dtype=float)
        return np.vstack([f(times) for f in sfns])

    def score(self, X, y):
        """Calculate concordance index using scikit-survival's scoring.

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        return self.model.score(X, y)


@dataclass
class CoxnetWrapper(BaseSurvivalModel):
    """Wrapper for elastic net regularized Cox model (Coxnet).

    Implements Cox PH with elastic net penalty (L1 + L2 regularization)
    using scikit-survival's CoxnetSurvivalAnalysis.

    Attributes:
        name: Model identifier, defaults to "coxnet"
        alphas: Regularization strengths to try. If None, auto-generated
        l1_ratio: Balance between L1 (1.0) and L2 (0.0) penalty. Defaults to 0.5
        alpha_min_ratio: Ratio of smallest to largest alpha in path
        n_alphas: Number of alphas in regularization path
        model: Underlying CoxnetSurvivalAnalysis instance
    """
    name: str = "coxnet"
    alphas: Optional[np.ndarray] = None
    l1_ratio: float = 0.5
    alpha_min_ratio: float = 0.01
    n_alphas: int = 100
    model: CoxnetSurvivalAnalysis = None

    def __post_init__(self):
        """Initialize Coxnet model with specified regularization parameters."""
        if self.model is None:
            self.model = CoxnetSurvivalAnalysis(
                l1_ratio=self.l1_ratio,
                alphas=self.alphas,
                alpha_min_ratio=self.alpha_min_ratio,
                n_alphas=self.n_alphas,
                max_iter=10_000,
                fit_baseline_model=True  # Required for predict_survival_function
            )

    def fit(self, X, y):
        """Fit elastic net Cox model.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance
        """
        self.model.fit(X, y)
        return self

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities
        """
        sfns = self.model.predict_survival_function(X)
        times = np.asarray(list(times), dtype=float)
        return np.vstack([f(times) for f in sfns])

    def score(self, X, y):
        """Calculate concordance index.

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        return self.model.score(X, y)


class StratifiedCoxWrapper(BaseSurvivalModel):
    """Wrapper for stratified Cox PH model using lifelines.

    Implements Cox PH with stratification on categorical variables to handle
    non-proportional hazards. Requires X to include original categorical columns
    (not one-hot encoded).

    Attributes:
        name: Model identifier, defaults to "cox_stratified"
        strata_cols: Tuple of column names to stratify on (from DataConfig)
        cph: Underlying CoxPHFitter instance from lifelines

    Note:
        This model expects DataFrame input with original categorical columns intact,
        not the transformed output from preprocessing pipeline.

    Example:
        >>> from survival_framework.config import DataConfig
        >>> config = DataConfig()
        >>> model = StratifiedCoxWrapper(strata_cols=config.stratification_columns)
    """

    def __init__(self, strata_cols: tuple[str, ...] = None):
        """Initialize stratified Cox model.

        Args:
            strata_cols: Tuple of column names to stratify on. If None, defaults
                to ("typeoftariff_coarse", "risk_level_coarse") for backward compatibility.
        """
        if strata_cols is None:
            # Backward compatibility default
            strata_cols = ("typeoftariff_coarse", "risk_level_coarse")

        self.name = "cox_stratified"
        self.strata_cols = strata_cols
        self.cph = CoxPHFitter()

    def fit(self, X_df: pd.DataFrame, y_struct):
        """Fit stratified Cox model.

        Args:
            X_df: DataFrame containing features including strata_cols
            y_struct: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance

        Raises:
            TypeError: If X_df is not a DataFrame (e.g., after VarianceThreshold)
            KeyError: If strata_cols not found in X_df
        """
        # Check if we received a numpy array instead of DataFrame (from pipeline)
        if not isinstance(X_df, pd.DataFrame):
            raise TypeError(
                f"StratifiedCoxWrapper requires DataFrame input with strata columns "
                f"{self.strata_cols}, but received {type(X_df).__name__}. "
                f"This model is incompatible with preprocessing pipelines that convert "
                f"to numpy arrays (e.g., VarianceThreshold). Use this model directly "
                f"with raw data, or skip it in pipeline-based training."
            )

        # Check if strata columns exist
        missing_cols = [col for col in self.strata_cols if col not in X_df.columns]
        if missing_cols:
            raise KeyError(
                f"Strata columns {missing_cols} not found in input DataFrame. "
                f"Available columns: {list(X_df.columns)[:10]}... "
                f"This model requires original categorical columns, not one-hot encoded features."
            )

        df = X_df.copy()
        df["time"] = y_struct["time"]
        df["event"] = y_struct["event"].astype(int)
        self.cph.fit(df, duration_col="time", event_col="event", strata=list(self.strata_cols))
        return self

    def predict_survival_function(self, X_df: pd.DataFrame, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X_df: DataFrame with features including strata_cols
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities
        """
        times = np.asarray(list(times), dtype=float)
        sf = self.cph.predict_survival_function(X_df, times=times)
        return sf.T.values

    def score(self, X_df, y_struct):
        """Calculate concordance index using lifelines.

        Args:
            X_df: DataFrame with features
            y_struct: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        df = X_df.copy()
        df["time"] = y_struct["time"]
        df["event"] = y_struct["event"].astype(int)
        return self.cph.concordance_index_(df)


@dataclass
class WeibullAFTWrapper(BaseSurvivalModel):
    """Wrapper for Weibull Accelerated Failure Time (AFT) model.

    Implements parametric survival model assuming Weibull distribution for
    survival times using lifelines' WeibullAFTFitter.

    Attributes:
        name: Model identifier, defaults to "weibull_aft"
        aft: Underlying WeibullAFTFitter instance from lifelines

    Note:
        AFT models provide interpretable acceleration factors showing how
        covariates speed up or slow down time to event.
    """
    name: str = "weibull_aft"
    aft: WeibullAFTFitter = None

    def __post_init__(self):
        """Initialize lifelines WeibullAFTFitter if not provided."""
        if self.aft is None:
            self.aft = WeibullAFTFitter()

    def fit(self, X_df, y_struct):
        """Fit Weibull AFT model.

        Args:
            X_df: DataFrame or array containing features (converted to DataFrame if needed)
            y_struct: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance

        Note:
            If X_df is a numpy array (from preprocessing pipeline), it's converted
            to a DataFrame with generic column names since lifelines requires DataFrames.
        """
        # Convert to DataFrame if needed (after preprocessing pipeline)
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=[f"X{i}" for i in range(X_df.shape[1])])

        df = X_df.copy()
        df["time"] = y_struct["time"]
        df["event"] = y_struct["event"].astype(int)

        # Weibull AFT requires strictly positive durations
        # Add small constant to zero/negative times
        min_time = df["time"].min()
        if min_time <= 0:
            df["time"] = df["time"] + abs(min_time) + 0.01

        self.aft.fit(df, duration_col="time", event_col="event")
        return self

    def predict_survival_function(self, X_df, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X_df: DataFrame or array with features (converted if needed)
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities
        """
        # Convert to DataFrame if needed (from preprocessing pipeline)
        if not isinstance(X_df, pd.DataFrame):
            X_df = pd.DataFrame(X_df, columns=[f"X{i}" for i in range(X_df.shape[1])])

        times = np.asarray(list(times), dtype=float)
        sf = self.aft.predict_survival_function(X_df, times=times)
        return sf.T.values

    def score(self, X_df, y_struct):
        """Calculate concordance index using lifelines.

        Args:
            X_df: DataFrame with features
            y_struct: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        df = X_df.copy()
        df["time"] = y_struct["time"]
        df["event"] = y_struct["event"].astype(int)
        return self.aft.concordance_index_(df)


@dataclass
class GBSAWrapper(BaseSurvivalModel):
    """Wrapper for Gradient Boosting Survival Analysis.

    Implements ensemble survival model using gradient boosting with
    scikit-survival's GradientBoostingSurvivalAnalysis. Builds trees
    sequentially to minimize survival loss.

    Attributes:
        name: Model identifier, defaults to "gbsa"
        model: Underlying GradientBoostingSurvivalAnalysis instance
        n_estimators: Number of boosting iterations
        learning_rate: Learning rate (shrinkage parameter)
        max_depth: Maximum depth of individual trees
        subsample: Fraction of samples to use for each tree
        min_samples_split: Minimum samples required to split a node

    Note:
        Gradient boosting often provides high predictive accuracy but may
        be prone to overfitting without proper regularization.
    """
    name: str = "gbsa"
    model: GradientBoostingSurvivalAnalysis = None
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    subsample: float = 1.0
    min_samples_split: int = 2

    def __post_init__(self):
        """Initialize GradientBoostingSurvivalAnalysis with configured parameters."""
        if self.model is None:
            self.model = GradientBoostingSurvivalAnalysis(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                min_samples_split=self.min_samples_split,
                random_state=42
            )

    def fit(self, X, y):
        """Fit gradient boosting survival model.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance
        """
        self.model.fit(X, y)
        return self

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities
        """
        sfns = self.model.predict_survival_function(X)
        times = np.asarray(list(times), dtype=float)
        return np.vstack([f(times) for f in sfns])

    def score(self, X, y):
        """Calculate concordance index.

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        return self.model.score(X, y)


@dataclass
class RSFWrapper(BaseSurvivalModel):
    """Wrapper for Random Survival Forest.

    Implements ensemble survival model using random forest approach with
    scikit-survival's RandomSurvivalForest. Averages predictions from
    multiple survival trees built on bootstrap samples.

    Attributes:
        name: Model identifier, defaults to "rsf"
        model: Underlying RandomSurvivalForest instance
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in leaf node
        max_features: Number of features to consider per split (None = sqrt(n_features))

    Note:
        RSF provides robust non-parametric estimates and handles non-linear
        relationships well. Returns step functions aligned to training event times.
    """
    name: str = "rsf"
    model: RandomSurvivalForest = None
    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Optional[int] = None

    def __post_init__(self):
        """Initialize RandomSurvivalForest with configured parameters."""
        if self.model is None:
            self.model = RandomSurvivalForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=42
            )

    def fit(self, X, y):
        """Fit random survival forest.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            self: Fitted model instance
        """
        self.model.fit(X, y)
        return self

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival probabilities at specified times.

        Args:
            X: Feature matrix with shape (n_samples, n_features)
            times: Time points for evaluation

        Returns:
            Array with shape (n_samples, n_times) containing survival probabilities

        Note:
            RSF returns step functions aligned to training event times which are
            then evaluated at the requested time points.
        """
        sfns = self.model.predict_survival_function(X)
        times = np.asarray(list(times), dtype=float)
        return np.vstack([f(times) for f in sfns])

    def score(self, X, y):
        """Calculate concordance index.

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Returns:
            Concordance index score
        """
        return self.model.score(X, y)


# Placeholder for DeepSurv integration
class DeepSurvWrapper(BaseSurvivalModel):
    """Placeholder wrapper for DeepSurv neural network model.

    Future implementation will integrate deep learning survival models
    using PyTorch and pycox library for neural network-based survival analysis.

    Attributes:
        name: Model identifier, "deepsurv"
        kwargs: Configuration parameters for future implementation

    Note:
        This is a placeholder. Implementation requires pycox and PyTorch dependencies.
        DeepSurv uses feed-forward neural networks to estimate Cox PH hazard functions.
    """
    name: str = "deepsurv"

    def __init__(self, **kwargs):
        """Initialize DeepSurv wrapper with configuration parameters.

        Args:
            **kwargs: Configuration parameters for future DeepSurv implementation
        """
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit DeepSurv model (not yet implemented).

        Args:
            X: Feature matrix
            y: Structured array with dtype=[('event', bool), ('time', float)]

        Raises:
            NotImplementedError: DeepSurv integration pending
        """
        raise NotImplementedError("DeepSurv integration pending (pycox/torch)")

    def predict_survival_function(self, X, times: Iterable[float]):
        """Predict survival function (not yet implemented).

        Args:
            X: Feature matrix
            times: Time points for evaluation

        Raises:
            NotImplementedError: DeepSurv integration pending
        """
        raise NotImplementedError

    def score(self, X, y):
        """Calculate concordance index (not yet implemented).

        Args:
            X: Feature matrix
            y: Structured array

        Raises:
            NotImplementedError: DeepSurv integration pending
        """
        raise NotImplementedError