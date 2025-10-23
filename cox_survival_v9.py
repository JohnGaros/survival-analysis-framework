#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_cox_survival.py

Single-entry production-ready script to train the best Cox model on tabular data using scikit-survival,
evaluate with CV (C-index, Brier/IBS), check PH via time-interaction surrogate, refit the best pipeline,
and output survival probabilities for months 1..36 for each id.

Run:
    python train_cox_survival.py \
        --data-path data/input.parquet \
        --output-dir ./artifacts \
        --random-seed 42

If --data-path is omitted, a synthetic dataset (~2k rows) is generated to validate the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import functools

# ---- Fail fast on missing critical dependency ----
try:
    import sksurv  # noqa: F401
except Exception as e:
    sys.stderr.write(
        "ERROR: scikit-survival is required. Install with `pip install scikit-survival`.\n"
        f"Original error: {repr(e)}\n"
    )
    sys.exit(1)

# Standard libs
import math
import json
import logging
import random
from functools import wraps
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

# Third-party scientific stack
import numpy as np
import pandas as pd
from joblib import dump
from lifelines import CoxPHFitter


# scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.base import clone

# scikit-survival
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

# Matplotlib (optional for diagnostics)
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Optional YAML config (not required by default)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None


# ========================================
# Timing utility - centralized execution time measurement
# ========================================

def time_function(func):
    """Decorator that logs execution time for any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f"{func.__name__} completed in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} (Hours:Minutes:Seconds)")
        return result
    return wrapper


# ========================================
# Constants & schema (unchanged)
# ========================================

@dataclass
class Cols:
    id_col: str = 'account_entities_key'
    duration_col: str = 'survival_months'
    event_col: str = 'is_terminated'
    tariff_col: str = 'typeoftariff'
    tariff_coarse_col: str = 'typeoftariff_coarse'
    risk_level_col: str = 'risk_level'
    risk_level_coarse_col: str = 'risk_level_coarse'
    debit_exp_smooth_col: str = 'debit_exp_smooth'
    credit_exp_smooth_col: str = 'credit_exp_smooth'
    balance_exp_smooth_col: str = 'balance_exp_smooth'
    past_due_balance_exp_smooth_col: str = 'past_due_balance_exp_smooth'
    total_settlements_col: str = 'total_settlements'
    active_settlements_col: str = 'active_settlements'
    defaulted_settlements_col: str = 'defaulted_settlements'

c = Cols()

ALL_COLUMNS = [
    c.id_col,
    c.duration_col,
    c.event_col,
    c.risk_level_coarse_col,
    c.debit_exp_smooth_col,
    c.credit_exp_smooth_col,
    c.balance_exp_smooth_col,
    c.past_due_balance_exp_smooth_col,
    c.total_settlements_col,
    c.active_settlements_col,
    c.defaulted_settlements_col,
    c.tariff_coarse_col,
]

CATEGORICAL_COLS = [c.risk_level_coarse_col, c.tariff_coarse_col]
NUMERIC_COLS = [
    c.debit_exp_smooth_col,
    c.credit_exp_smooth_col,
    c.balance_exp_smooth_col,
    c.past_due_balance_exp_smooth_col,
    c.total_settlements_col,
    c.active_settlements_col,
    c.defaulted_settlements_col,
]

ID_COL = c.id_col
DURATION_COL = c.duration_col
EVENT_COL = c.event_col

PREDICTION_MONTHS = list(range(1, 37))  # 1..36 inclusive


# ========================================
# All existing utility and transformer classes remain unchanged
# ========================================

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def censoring_rate(events: np.ndarray) -> float:
    return float(np.mean(events == 0))


def assert_prediction_monotonicity(df_wide: pd.DataFrame) -> None:
    cols = [f"S_{m}m" for m in PREDICTION_MONTHS]
    probs = df_wide[cols].to_numpy()
    # in [0,1]
    if not (np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()):
        raise ValueError("Predicted survival probabilities must be within [0, 1].")
    # non-increasing over time per id
    diffs = np.diff(probs, axis=1)
    if (diffs > 1e-8).any():
        raise ValueError("Survival probabilities must be non-increasing over time per id.")


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


# [All transformer classes remain unchanged - RareCategoryGrouper, ImputeWithIndicator, 
#  UnivariateCoxSelector, AddLogTimeInteractions, etc.]

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups rare categories to 'Other' for each categorical feature independently.
    Frequency threshold is a proportion in (0,1). Levels with freq < threshold are mapped to 'Other'.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.level_maps_: Dict[str, Dict[Any, Any]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        self.columns_ = list(X.columns)
        self.level_maps_ = {}
        n = len(X)
        for col in self.columns_:
            vc = X[col].value_counts(dropna=False)
            keep = set(vc[vc / n >= self.threshold].index.tolist())
            m: Dict[Any, Any] = {}
            for level in vc.index:
                if level in keep:
                    m[level] = level
                else:
                    m[level] = "Other"
            # Ensure "Other" is stable
            self.level_maps_[col] = m
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns_:
            m = self.level_maps_[col]
            X[col] = X[col].map(lambda v: m.get(v, "Other"))
        return X


class ImputeWithIndicator(BaseEstimator, TransformerMixin):
    """
    Impute columns with provided strategy and append missing-indicator columns (one per input feature).
    """

    def __init__(self, strategy: str = "median", fill_value: Optional[Any] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer_: Optional[SimpleImputer] = None
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        self.columns_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value, add_indicator=True)
        self.imputer_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise RuntimeError("ImputeWithIndicator not fitted.")
        arr = self.imputer_.transform(X)  # numeric ndarray
        # Build column names: original + indicator columns
        n_features = len(self.columns_)
        n_total = arr.shape[1]
        n_ind = n_total - n_features
        cols = self.columns_ + [f"{c}_missing" for c in self.columns_][:n_ind]
        return pd.DataFrame(arr, columns=cols, index=X.index)



class UnivariateCoxSelector(BaseEstimator, TransformerMixin):
    """
    Univariate feature screening using lifelines CoxPHFitter (one feature at a time).
    Ranks features by Harrell's C-index and filters by p-value.

    Parameters
    ----------
    k : int or 'all'
        Number of top features to select. If 'all', keep all.
    pval_threshold : float
        Maximum p-value to consider a feature statistically significant.
    min_features : int
        Minimum number of features to select.
    """

    def __init__(self, k: Any = "all", pval_threshold: float = 0.05, min_features: int = 5):
        self.k = k
        self.pval_threshold = pval_threshold
        self.min_features = min_features
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # y expected as dtype=[('event', bool), ('time', float)]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        scores: List[Tuple[str, float, float]] = []
        # Prepare y for lifelines
        y_df = pd.DataFrame({"duration": y["time"], "event": y["event"].astype(int)})
        for col in X.columns:
            try:
                df = pd.DataFrame({col: X[col], "duration": y_df["duration"], "event": y_df["event"]})
                cph = CoxPHFitter()
                cph.fit(df, duration_col="duration", event_col="event")
                coef = cph.params_[col]
                pval = cph.summary.loc[col, "p"]
                # Use lifelines concordance_index_ for C-index
                c_index = cph.concordance_index_
                scores.append((col, float(c_index), float(pval)))
            except Exception:
                # If a feature fails, score as 0.5 (random), p-value as 1.0 (not significant)
                scores.append((col, 0.5, 1.0))
        # Filter by p-value threshold
        filtered = [t for t in scores if t[2] <= self.pval_threshold]
        # If not enough features, relax p-value filter
        if len(filtered) < self.min_features:
            filtered = scores
        # Sort by C-index descending
        filtered.sort(key=lambda t: t[1], reverse=True)
        if self.k == "all":
            ksel = len(filtered)
        else:
            ksel = max(int(self.k), self.min_features)
        self.selected_features_ = [name for name, _, _ in filtered[:ksel]]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not self.selected_features_:
            return X
        return X[self.selected_features_]
    

class AddLogTimeInteractions(BaseEstimator, TransformerMixin):
    """
    Adds interaction features: each input feature multiplied by log(time + 1).

    Note:
        Requires access to y (Surv array) in fit/transform to compute log(time+1).
        We cache indices to align rows. Used inside a Pipeline where fit/transform
        are called with y provided.

    Parameters
    ----------
    feature_names : Optional[List[str]]
        If provided, subset of columns to interact; otherwise, interact all numeric features.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.feature_names is None:
            # Heuristic: choose numeric columns
            self.columns_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        else:
            self.columns_ = list(self.feature_names)
        return self

    def transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Ensure all column names are strings
        X.columns = X.columns.astype(str)
        
        if y is None:
            # At prediction time, we do not know future times; use a neutral factor of 1.0 (log(0+1)=0)
            # So interaction terms become zero -> no effect on survival at reference baseline time 0.
            logt = np.zeros(X.shape[0], dtype=float)
        else:
            logt = np.log1p(y["time"].astype(float))
        X = X.copy()
        for col in self.columns_:
            col_str = str(col)  # Ensure column name is string
            X[f"{col_str}__x__logt"] = X[col_str].to_numpy(dtype=float) * logt
        return X


class AddStrataInteractions(BaseEstimator, TransformerMixin):
    """
    Add interaction terms between all features and stratification variables.
    Applied BEFORE preprocessing to work with original categorical data.
    """
    
    def __init__(self, strata_cols: List[str]):
        self.strata_cols = strata_cols
        self.feature_cols_: List[str] = []
        self.strata_levels_: Dict[str, List[str]] = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Identify non-strata features
        self.feature_cols_ = [col for col in X.columns if col not in self.strata_cols]
        
        # Get unique levels for each strata variable
        self.strata_levels_ = {}
        for strata_col in self.strata_cols:
            if strata_col in X.columns:
                unique_vals = X[strata_col].fillna("Missing").unique()
                self.strata_levels_[strata_col] = list(unique_vals)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Pre-calculate all interaction data as numpy arrays
        interaction_data = {}
        
        for feature_col in self.feature_cols_:
            if feature_col not in X.columns:
                continue
                
            feature_values = X[feature_col].values  # Get numpy array
            
            for strata_col, strata_levels in self.strata_levels_.items():
                if strata_col not in X.columns:
                    continue
                    
                strata_values = X[strata_col].fillna("Missing").values
                
                for strata_val in strata_levels:
                    # Create indicator and interaction in one step
                    indicator = (strata_values == strata_val).astype(int)
                    interaction_name = f"{feature_col}__x__{strata_col}={strata_val}"
                    interaction_data[interaction_name] = feature_values * indicator
        
        # Create DataFrame from all interactions at once
        if interaction_data:
            interaction_df = pd.DataFrame(interaction_data, index=X.index)
            result = pd.concat([X, interaction_df], axis=1)
        else:
            result = X.copy()
        
        return result


# ...existing code...

class StratifiedModelTrainer:
    """
    Decorator that enables training distinct survival models for each unique combination
    of stratification variables (tariff_coarse_col and risk_level_coarse_col).
    
    Can be applied to any model building function to automatically handle
    stratified training, prediction, and evaluation.
    """
    
    def __init__(self, strata_cols: List[str] = None):
        self.strata_cols = strata_cols or [c.tariff_coarse_col, c.risk_level_coarse_col]
        
    def __call__(self, model_builder_func):
        """Decorator that wraps model building functions to enable stratified training."""
        @wraps(model_builder_func)
        def wrapper(*args, **kwargs):
            # Create a factory function that returns a properly configured StratifiedPipeline
            def stratified_pipeline_factory():
                return StratifiedPipeline(
                    base_model_builder=model_builder_func,
                    base_args=args,
                    base_kwargs=kwargs,
                    strata_cols=self.strata_cols
                )
            
            # Get the base pipeline and param grid for parameter structure
            base_pipe, param_grid = model_builder_func(*args, **kwargs)
            
            # Return the factory-created pipeline
            stratified_pipe = stratified_pipeline_factory()
            
            return stratified_pipe, param_grid
            
        return wrapper


class StratifiedPipeline(BaseEstimator):
    """
    Pipeline wrapper that trains separate models for each stratum combination.
    Maintains sklearn Pipeline interface for compatibility with GridSearchCV.
    """
    
    def __init__(
        self,
        base_model_builder=None,
        base_args=None,
        base_kwargs=None,
        strata_cols=None,
        **kwargs  # Accept any extra parameters for sklearn compatibility
    ):
        # Store our core parameters
        self.base_model_builder = base_model_builder
        self.base_args = base_args if base_args is not None else []
        self.base_kwargs = base_kwargs if base_kwargs is not None else {}
        self.strata_cols = strata_cols if strata_cols is not None else []
        
        # Initialize internal state
        self.strata_models_: Dict[str, Pipeline] = {}
        self.strata_mapping_: Dict[str, str] = {}
        self.fitted_ = False
        
        # Store ALL extra parameters as attributes for sklearn compatibility
        # This is crucial for clone() to work properly
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for GridSearchCV compatibility."""
        # Start with our core parameters
        params = {
            'base_model_builder': self.base_model_builder,
            'base_args': self.base_args,
            'base_kwargs': self.base_kwargs,
            'strata_cols': self.strata_cols,
        }
        
        # Add all attributes that don't start with underscore and aren't methods
        for attr_name in dir(self):
            if (not attr_name.startswith('_') and 
                not callable(getattr(self, attr_name)) and
                attr_name not in params):
                try:
                    attr_value = getattr(self, attr_name)
                    # Only include serializable types
                    if isinstance(attr_value, (str, int, float, bool, list, tuple, dict, type(None))):
                        params[attr_name] = attr_value
                except Exception:
                    pass
        
        return params
    
    def set_params(self, **params) -> 'StratifiedPipeline':
        """Set parameters for GridSearchCV compatibility."""
        # Set any parameter as an attribute
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def _get_strata_key(self, X: pd.DataFrame, idx: int) -> str:
        """Generate stratum key for a given row."""
        key_parts = []
        for col in self.strata_cols:
            if col in X.columns:
                val = str(X.iloc[idx][col])
            else:
                val = "Missing"
            key_parts.append(f"{col}={val}")
        return "__".join(key_parts)
    
    def _build_strata_mapping(self, X: pd.DataFrame) -> None:
        """Build mapping from row indices to strata keys."""
        self.strata_mapping_ = {}
        for idx in range(len(X)):
            self.strata_mapping_[idx] = self._get_strata_key(X, idx)
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit separate models for each stratum."""
        # Defensive: check required attributes
        if (self.base_model_builder is None or not self.strata_cols):
            raise ValueError(
                "StratifiedPipeline is missing required initialization arguments. "
                "This usually happens if sklearn.clone() is called without proper parameters."
            )
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self._build_strata_mapping(X)
        
        # Group data by strata
        strata_data = defaultdict(list)
        for idx, strata_key in self.strata_mapping_.items():
            strata_data[strata_key].append(idx)
        
        self.strata_models_ = {}
        
        for strata_key, indices in strata_data.items():
            # More lenient threshold since we've pre-filtered thin strata
            if len(indices) < 5:  # Reduced from 10 to 5
                logging.warning(f"Skipping stratum {strata_key}: only {len(indices)} observations (this should be rare after pre-filtering)")
                continue
                
            # Extract stratum data
            X_stratum = X.iloc[indices].copy()
            y_stratum = y[indices]
            
            # Check if we have any events in this stratum
            if not np.any(y_stratum["event"]):
                logging.warning(f"Skipping stratum {strata_key}: no events observed")
                continue
            
            try:
                # Build and fit model for this stratum
                base_pipe, _ = self.base_model_builder(*self.base_args, **self.base_kwargs)
                fitted_pipe = base_pipe.fit(X_stratum, y_stratum)
                self.strata_models_[strata_key] = fitted_pipe
                
                logging.info(f"Fitted model for stratum {strata_key}: {len(indices)} observations, "
                           f"{np.sum(y_stratum['event'])} events")
                           
            except Exception as e:
                logging.warning(f"Failed to fit model for stratum {strata_key}: {e}")
                continue
        
        if not self.strata_models_:
            raise ValueError("No valid strata models could be fitted")
            
        self.fitted_ = True
        return self
    
    # ... rest of the StratifiedPipeline methods remain the same ...
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using appropriate stratum model for each observation."""
        if not self.fitted_:
            raise ValueError("StratifiedPipeline not fitted")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        predictions = np.full(len(X), np.nan)
        
        for idx in range(len(X)):
            strata_key = self._get_strata_key(X, idx)
            
            if strata_key in self.strata_models_:
                model = self.strata_models_[strata_key]
                try:
                    pred = model.predict(X.iloc[[idx]])
                    predictions[idx] = pred[0]
                except Exception as e:
                    logging.warning(f"Prediction failed for stratum {strata_key}: {e}")
                    # Fall back to first available model
                    fallback_model = next(iter(self.strata_models_.values()))
                    try:
                        pred = fallback_model.predict(X.iloc[[idx]])
                        predictions[idx] = pred[0]
                    except Exception:
                        predictions[idx] = 0.0  # Neutral risk score
            else:
                # Use first available model as fallback
                if self.strata_models_:
                    fallback_model = next(iter(self.strata_models_.values()))
                    try:
                        pred = fallback_model.predict(X.iloc[[idx]])
                        predictions[idx] = pred[0]
                    except Exception:
                        predictions[idx] = 0.0
                        
        return predictions
    
    def predict_survival_function(self, X: pd.DataFrame, return_array: bool = True):
        """Predict survival functions using appropriate stratum model."""
        if not self.fitted_:
            raise ValueError("StratifiedPipeline not fitted")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        survival_functions = []
        
        for idx in range(len(X)):
            strata_key = self._get_strata_key(X, idx)
            
            if strata_key in self.strata_models_:
                model = self.strata_models_[strata_key]
                try:
                    sf = model.predict_survival_function(X.iloc[[idx]], return_array=return_array)
                    survival_functions.append(sf[0])
                except Exception as e:
                    logging.warning(f"Survival function prediction failed for stratum {strata_key}: {e}")
                    # Fall back to first available model
                    fallback_model = next(iter(self.strata_models_.values()))
                    sf = fallback_model.predict_survival_function(X.iloc[[idx]], return_array=return_array)
                    survival_functions.append(sf[0])
            else:
                # Use first available model as fallback
                if self.strata_models_:
                    fallback_model = next(iter(self.strata_models_.values()))
                    sf = fallback_model.predict_survival_function(X.iloc[[idx]], return_array=return_array)
                    survival_functions.append(sf[0])
                        
        return survival_functions


# Create decorator instances for each model type
stratified_trainer = StratifiedModelTrainer()


def c_index_scorer_func(estimator, X, y) -> float:
    """Harrell's C-index on risk scores (higher risk = shorter survival)."""
    try:
        risk = estimator.predict(X)
    except Exception:
        # Some estimators (e.g., Coxnet) use decision_function as risk
        risk = estimator.decision_function(X)
    
    # Use risk directly (not -risk)
    c_index = concordance_index_censored(y["event"], y["time"], risk)[0]
    return float(c_index)


c_index_scorer = make_scorer(c_index_scorer_func, greater_is_better=True,  needs_proba=False)


def compute_ibs(
    estimator, X: pd.DataFrame, y: np.ndarray, times: np.ndarray
) -> float:
    """Integrated Brier Score with Kaplan-Meier reference."""
    surv_funcs = estimator.predict_survival_function(X)
    # Convert to array at times
    preds = np.vstack([sf(times) for sf in surv_funcs])
    # brier_score returns (times, scores) - fix parameter names
    _, bs = brier_score(survival_train=y, survival_test=y, estimate=preds, times=times)
    ibs = integrated_brier_score(survival_train=y, survival_test=y, estimate=preds, times=times)
    return float(ibs)


@time_function
def test_stratification_need(X_raw: pd.DataFrame, y: np.ndarray, 
                           strata_cols: List[str], 
                           alpha: float = 0.05) -> Tuple[bool, Dict[str, float]]:
    """
    Test whether stratification is needed using log-likelihood ratio tests.
    
    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw feature data
    y : np.ndarray
        Survival data (structured array)
    strata_cols : List[str]
        Columns to test for stratification
    alpha : float
        Significance level for the test
        
    Returns
    -------
    needs_stratification : bool
        Whether stratification is statistically justified
    test_results : Dict[str, float]
        P-values for each stratification test
    """
    from lifelines import CoxPHFitter
    from scipy import stats
    from sklearn.preprocessing import LabelEncoder
    
    test_results = {}
    needs_stratification = False
    
    for strata_col in strata_cols:
        if strata_col not in X_raw.columns:
            logging.warning(f"Stratification column {strata_col} not found in data")
            test_results[strata_col] = 1.0
            continue
            
        try:
            # Create a minimal dataset with only the stratification column
            strata_series = X_raw[strata_col].copy()
            
            # Convert to string and handle missing values
            strata_series = strata_series.astype(str).replace(['nan', 'None', '', 'NaN'], 'Missing')
            
            # Check for obvious data quality issues
            sample_values = strata_series.dropna().unique()[:5]
            
            # Skip if values look like dates, IDs, or other non-categorical data
            suspicious_patterns = [
                any('-' in str(val) and len(str(val)) >= 8 for val in sample_values),  # Date-like
                any(len(str(val)) > 20 for val in sample_values),  # Very long strings
                len(strata_series.unique()) > min(50, len(X_raw) // 10)  # Too many categories
            ]
            
            if any(suspicious_patterns):
                logging.warning(f"Skipping stratification test for {strata_col}: data appears non-categorical (sample: {sample_values[:3]})")
                test_results[strata_col] = 1.0
                continue
            
            # Ensure we have enough observations per stratum for meaningful test
            value_counts = strata_series.value_counts()
            if len(value_counts) < 2:
                logging.warning(f"Skipping stratification test for {strata_col}: only {len(value_counts)} unique value(s)")
                test_results[strata_col] = 1.0
                continue
                
            # Filter out strata with very few observations
            min_obs_per_stratum = max(5, len(X_raw) // 100)
            valid_strata = value_counts[value_counts >= min_obs_per_stratum].index
            
            if len(valid_strata) < 2:
                logging.warning(f"Skipping stratification test for {strata_col}: insufficient observations per stratum")
                test_results[strata_col] = 1.0
                continue
            
            # Filter data to only include valid strata
            valid_mask = strata_series.isin(valid_strata)
            
            if valid_mask.sum() < 20:  # Need minimum sample size
                logging.warning(f"Skipping stratification test for {strata_col}: insufficient data after filtering")
                test_results[strata_col] = 1.0
                continue
            
            # **KEY FIX: Encode categorical variable as numeric for lifelines**
            strata_filtered = strata_series[valid_mask]
            
            # Use LabelEncoder to convert strings to integers
            label_encoder = LabelEncoder()
            strata_encoded = label_encoder.fit_transform(strata_filtered)
            
            # Create minimal test dataset with numeric stratification variable
            df_minimal = pd.DataFrame({
                "duration": y["time"][valid_mask].astype(float),
                "event": y["event"][valid_mask].astype(int),
                f"{strata_col}_encoded": strata_encoded.astype(int)  # Use numeric encoding
            })
            
            # Drop any rows with invalid data
            df_minimal = df_minimal.dropna()
            
            if len(df_minimal) < 20:
                logging.warning(f"Skipping stratification test for {strata_col}: insufficient valid data")
                test_results[strata_col] = 1.0
                continue
            
            # Log the encoding mapping for debugging
            encoding_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            logging.info(f"Stratification encoding for {strata_col}: {encoding_map}")
            
            # Fit unstratified model (no covariates, just baseline hazard)
            cph_unstratified = CoxPHFitter()
            cph_unstratified.fit(df_minimal, duration_col="duration", event_col="event")
            ll_unstratified = cph_unstratified.log_likelihood_
            
            # Fit stratified model using the encoded numeric column
            cph_stratified = CoxPHFitter()
            cph_stratified.fit(df_minimal, duration_col="duration", event_col="event", 
                             strata=[f"{strata_col}_encoded"])
            ll_stratified = cph_stratified.log_likelihood_
            
            # Likelihood ratio test
            lr_stat = 2 * (ll_stratified - ll_unstratified)
            
            # Degrees of freedom = (number of strata - 1)
            n_strata = df_minimal[f"{strata_col}_encoded"].nunique()
            df_lr = n_strata - 1
            
            # P-value from chi-square distribution
            if lr_stat < 0 or df_lr <= 0:
                logging.warning(f"Invalid likelihood ratio test for {strata_col}: LR={lr_stat}, df={df_lr}")
                p_value = 1.0
            else:
                p_value = 1 - stats.chi2.cdf(lr_stat, df_lr)
            
            test_results[strata_col] = p_value
            
            if p_value < alpha:
                needs_stratification = True
                logging.info(f"Stratification test for {strata_col}: p={p_value:.4f} (significant)")
            else:
                logging.info(f"Stratification test for {strata_col}: p={p_value:.4f} (not significant)")
                
        except Exception as e:
            logging.warning(f"Stratification test failed for {strata_col}: {e}")
            test_results[strata_col] = 1.0  # Conservative: assume no stratification needed
    
    logging.info(f"Overall stratification needed: {needs_stratification}")
    return needs_stratification, test_results


def build_preprocessor(rare_threshold: float = 0.01) -> Pipeline:
    """
    ColumnTransformer that:
        - groups rare categorical levels
        - imputes (with missing indicators)
        - one-hot encodes categoricals (drop='first')
        - standardizes numeric features
        - drops near-zero variance features
    """
    # Categorical branch
    cat_pipe = Pipeline(
        steps=[
            ("rare", RareCategoryGrouper(threshold=rare_threshold)),
            ("impute", ImputeWithIndicator(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # Numeric branch
    num_pipe = Pipeline(
        steps=[
            ("impute", ImputeWithIndicator(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_COLS),
            ("cat", cat_pipe, CATEGORICAL_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,  # Use simple feature names
    )

    # After concatenation, drop near-constant columns
    full = Pipeline(
        steps=[
            ("pre", pre),
            ("varth", VarianceThreshold(threshold=1e-12)),
        ]
    )
    
    return full


def build_model_A(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """
    Model A: Preprocessing -> UnivariateCoxSelector -> CoxPH
    """
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("uv", UnivariateCoxSelector(k="all")),
            ("coxph", CoxPHSurvivalAnalysis()),
        ]
    )
    param_grid = {"uv__k": k_values}
    return pipe, param_grid


def build_model_B() -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """
    Model B: Preprocessing -> Coxnet (elastic net)
    """
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("coxnet", CoxnetSurvivalAnalysis(
                l1_ratio=0.5, 
                alpha_min_ratio="auto", 
                fit_baseline_model=True  # Required for predict_survival_function
            )),
        ]
    )
    # Use auto alphas via algorithm; tune l1_ratio
    param_grid = {"coxnet__l1_ratio": [0.01, 0.25, 0.5, 0.75, 1.0]}
    return pipe, param_grid


def build_model_C(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """
    Model C: Preprocessing -> AddLogTimeInteractions -> UnivariateCoxSelector -> CoxPH

    Surrogate for PH diagnostics: include multiplicative log(time) interactions.
    """
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("logt", AddLogTimeInteractions()),
            ("uv", UnivariateCoxSelector(k="all")),
            ("coxph", CoxPHSurvivalAnalysis(alpha=1)),  # Add L2 regularization
        ]
    )
    param_grid = {"uv__k": [5, 10, 20]}  # Use fewer features
    return pipe, param_grid


def build_model_D(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """
    Model D: Stratified Cox model simulation via interaction terms
    
    Simulates stratification by creating interaction terms between all features 
    and the stratification variables (tariff_coarse_col, risk_level_coarse_col).
    This allows different baseline hazards and feature effects for each stratum.
    """
    # Apply strata interactions BEFORE preprocessing
    pipe = Pipeline(
        steps=[
            ("strata", AddStrataInteractions(strata_cols=[c.tariff_coarse_col, c.risk_level_coarse_col])),
            ("pre", build_preprocessor()),
            ("uv", UnivariateCoxSelector(k="all")),
            ("coxph", CoxPHSurvivalAnalysis()),
        ]
    )
    param_grid = {"uv__k": k_values}
    return pipe, param_grid


# Modify the model building functions to support stratification
def build_stratified_models(k_values: List[Any]) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """
    Build stratified versions of all model types.
    Returns dictionary mapping model names to (pipeline, param_grid) tuples.
    """
    
    @stratified_trainer
    def stratified_model_A(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_A(k_values)
    
    @stratified_trainer  
    def stratified_model_B() -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_B()
    
    @stratified_trainer
    def stratified_model_C(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_C(k_values)
    
    @stratified_trainer
    def stratified_model_D(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_D(k_values)
    
    # Build all stratified models
    stratified_models = {}
    
    model_A_strat, grid_A_strat = stratified_model_A(k_values=k_values)
    stratified_models["ModelA_Stratified"] = (model_A_strat, grid_A_strat)
    
    model_B_strat, grid_B_strat = stratified_model_B()
    stratified_models["ModelB_Stratified"] = (model_B_strat, grid_B_strat)
    
    model_C_strat, grid_C_strat = stratified_model_C(k_values=k_values)
    stratified_models["ModelC_Stratified"] = (model_C_strat, grid_C_strat)
    
    model_D_strat, grid_D_strat = stratified_model_D(k_values=[5, 10, 20])
    stratified_models["ModelD_Stratified"] = (model_D_strat, grid_D_strat)
    
    return stratified_models


def plot_brier_curve(
    times: np.ndarray, brier_scores: np.ndarray, out_path: str, title: str
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(times, brier_scores, label="Brier score")
    plt.xlabel("Time (months)")
    plt.ylabel("Brier score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def calibration_plot_at_time(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    t: float,
    out_path: str,
    title: str,
    n_bins: int = 10,
) -> None:
    """
    Group by predicted survival at time t into deciles; compare predicted vs observed (KM) survival.
    """
    sfs = estimator.predict_survival_function(X)
    
    # Fix: Handle both array and scalar returns from survival functions
    preds = []
    for sf in sfs:
        try:
            # Try to get survival probability at time t
            surv_at_t = sf(t)
            # Handle both scalar and array returns
            if np.isscalar(surv_at_t):
                preds.append(float(surv_at_t))
            else:
                # If it's an array, take the first (and likely only) element
                preds.append(float(surv_at_t[0]) if len(surv_at_t) > 0 else 1.0)
        except (IndexError, TypeError, ValueError):
            # Fallback to 1.0 (no event) if we can't evaluate at time t
            preds.append(1.0)
    
    preds = np.array(preds)
    
    # Ensure we have valid predictions
    if len(preds) == 0 or np.all(np.isnan(preds)):
        raise ValueError(f"No valid survival predictions at time {t}")
    
    # Bin into deciles - handle edge case where all predictions are the same
    unique_preds = np.unique(preds)
    if len(unique_preds) == 1:
        # All predictions are the same - create single bin
        bins = np.zeros(len(preds), dtype=int)
        n_bins = 1
    else:
        quantiles = np.quantile(preds, np.linspace(0, 1, n_bins + 1))
        # Ensure quantiles are unique to avoid empty bins
        quantiles = np.unique(quantiles)
        if len(quantiles) <= 2:
            bins = np.zeros(len(preds), dtype=int)
            n_bins = 1
        else:
            bins = np.clip(np.digitize(preds, quantiles[1:-1], right=True), 0, len(quantiles) - 2)
            n_bins = len(quantiles) - 1

    obs = []
    pred = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        pred.append(np.mean(preds[mask]))
        
        # Observed KM at time t in this bin
        times_bin = y["time"][mask]
        events_bin = y["event"][mask]
        
        try:
            km_result = kaplan_meier_estimator(events_bin, times_bin)
            km_t, km_s = km_result[:2]
            
            # Interpolate KM at time t
            if len(km_t) == 0:
                km_val = 1.0
            else:
                # Find the last time point <= t
                valid_indices = km_t <= t
                if np.any(valid_indices):
                    km_val = float(km_s[valid_indices][-1])
                else:
                    km_val = 1.0  # No events before time t
            obs.append(km_val)
        except Exception:
            # If KM estimation fails, assume no events (survival = 1.0)
            obs.append(1.0)

    if len(pred) == 0 or len(obs) == 0:
        raise ValueError(f"No valid bins for calibration plot at time {t}")

    pred = np.array(pred)
    obs = np.array(obs)

    plt.figure(figsize=(6, 6))
    plt.scatter(pred, obs, s=30, alpha=0.8)
    lims = [0.0, 1.0]
    plt.plot(lims, lims, linestyle="--", color="red", alpha=0.7, label="Perfect calibration")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Predicted S(t)")
    plt.ylabel("Observed S(t) (KM)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@dataclass
class CVResult:
    name: str
    estimator: Pipeline
    grid: Dict[str, List[Any]]
    best_estimator_: Optional[Pipeline] = None
    best_params_: Optional[Dict[str, Any]] = None
    best_score_: float = float("-inf")
    cv_results_: Optional[pd.DataFrame] = None
    ibs_tiebreak_: Optional[float] = None


def run_grid_search(
    name: str,
    pipe: Pipeline,
    param_grid: Dict[str, List[Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> CVResult:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=c_index_scorer_func,
        cv=skf,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    gs.fit(X, y)
    results = pd.DataFrame(gs.cv_results_)
    return CVResult(
        name=name,
        estimator=pipe,
        grid=param_grid,
        best_estimator_=gs.best_estimator_,
        best_params_=gs.best_params_,
        best_score_=float(gs.best_score_),
        cv_results_=results,
        ibs_tiebreak_=None,
    )


def tiebreak_with_ibs(
    candidates: List[CVResult],
    X: pd.DataFrame,
    y: np.ndarray,
    times: np.ndarray,
    tol: float = 1e-8,
) -> CVResult:
    """
    Among candidates with highest mean C-index (within tol), compute IBS and choose lowest IBS (best).
    """
    if not candidates:
        raise ValueError("No candidates for tiebreak.")
    # Find max C-index
    max_c = max(c.best_score_ for c in candidates)
    tied = [c for c in candidates if (max_c - c.best_score_) <= tol]
    if len(tied) == 1:
        return tied[0]

    # Evaluate IBS for tied candidates
    for c in tied:
        est = c.best_estimator_
        if est is None:
            continue
        try:
            ibs = compute_ibs(est, X, y, times)
        except Exception:
            ibs = float("inf")
        c.ibs_tiebreak_ = ibs

    tied.sort(key=lambda r: (r.ibs_tiebreak_ if r.ibs_tiebreak_ is not None else float("inf")))
    return tied[0]


def read_data(path: Optional[str]) -> pd.DataFrame:
    if path is None:
        logging.info("No --data-path provided. Generating synthetic dataset (~2000 rows)...")
        return generate_synthetic_data(n=2000, seed=42)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use CSV, Parquet, or Pickle.")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    # Fail on completely missing critical columns
    if df[DURATION_COL].isna().all() or df[EVENT_COL].isna().all():
        raise ValueError("duration_col or event_col is completely missing.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    # Drop missing duration/event
    df = df.dropna(subset=[DURATION_COL, EVENT_COL])
    
    # Non-positive durations: drop with warning
    bad = df[DURATION_COL] <= 0
    if bad.any():
        logging.warning("Dropping %d rows with non-positive durations.", int(bad.sum()))
        df = df.loc[~bad].copy()
    
    # Ensure event is binary 0/1 int
    df[EVENT_COL] = df[EVENT_COL].astype(int)
    
    # Map Greek characters and other special characters in categorical columns
    greek_mapping = {
        'Γ1': 'Gamma1',
        'Γ2': 'Gamma2',
        'Γ3': 'Gamma3',  # Add more as needed
        'Α': 'A',
        'Β': 'B',
        'Δ': 'D',
        'Ε': 'E',
        'Ζ': 'Z',
        'Η': 'E',
        'Θ': 'Theta',
        'Ι': 'I',
        'Κ': 'K',
        'Λ': 'L',
        'Μ': 'M',
        'Ν': 'N',
        'Ξ': 'X',
        'Ο': 'O',
        'Π': 'P',
        'Ρ': 'R',
        'Σ': 'S',
        'Τ': 'T',
        'Υ': 'U',
        'Φ': 'Phi',
        'Χ': 'Chi',
        'Ψ': 'Psi',
        'Ω': 'Omega'
    }
    
    # Apply Greek character mapping to categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            # Replace Greek characters
            for greek_char, latin_name in greek_mapping.items():
                df[col] = df[col].astype(str).str.replace(greek_char, latin_name, regex=False)
    
    # Ensure categorical dtypes are strings (better for OneHot)
    for c_col in CATEGORICAL_COLS + [ID_COL]:
        if c_col in df.columns:
            df[c_col] = df[c_col].astype(str)
    
    return df


def build_y_structured(df: pd.DataFrame) -> np.ndarray:
    # Surv(event, time) expects event as boolean
    y = Surv.from_arrays(event=df[EVENT_COL].astype(bool), time=df[DURATION_COL].astype(float))
    return y


def get_feature_names_from_pipeline(p: Pipeline, X: pd.DataFrame, y: np.ndarray) -> List[str]:
    if not isinstance(p, Pipeline) or len(p.steps) < 2:
        return list(X.columns)

    preproc = Pipeline(p.steps[:-1])
    try:
        preproc.fit(X, y)
    except Exception:
        pass

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.feature_selection import VarianceThreshold

    def find_ct_and_index(pipe: Pipeline):
        for i, (_, step) in enumerate(pipe.steps):
            if isinstance(step, ColumnTransformer):
                return step, i
            if hasattr(step, "steps"):
                for __, sub in step.steps:
                    if isinstance(sub, ColumnTransformer):
                        return sub, i
        return None, None

    ct, ct_idx = find_ct_and_index(preproc)

    if ct is None:
        try:
            Xt = preproc.transform(X) if hasattr(preproc, "transform") else preproc.fit_transform(X, y)
            if hasattr(Xt, "columns"):
                return list(Xt.columns)
            return [f"feature_{i}" for i in range(Xt.shape[1])]
        except Exception:
            return list(X.columns)

    def reconstruct_ct_feature_names(ct: ColumnTransformer) -> List[str]:
        names: List[str] = []
        for _, trans, cols in ct.transformers_:
            if _ == "remainder" and trans == "drop":
                continue

            cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray)) else [cols]

            if trans == "passthrough":
                names.extend([str(c) for c in cols_list])
                continue

            if hasattr(trans, "steps"):
                current = [str(c) for c in cols_list]
                for __, step_obj in trans.steps:
                    if step_obj.__class__.__name__ == "ImputeWithIndicator":
                        try:
                            imp = getattr(step_obj, "imputer_", None)
                            ind = getattr(imp, "indicator_", None) if imp is not None else None
                            if ind is not None and hasattr(ind, "features_") and len(ind.features_) > 0:
                                miss_cols = [current[i] for i in ind.features_ if i < len(current)]
                                current = current + [f"{c}_missing" for c in miss_cols]
                        except Exception:
                            pass
                        continue

                    if isinstance(step_obj, OneHotEncoder):
                        try:
                            cats = step_obj.categories_
                            drop = getattr(step_obj, "drop", None)
                            new_cols = []
                            for c_name, c_cats in zip(current, cats):
                                dropped = 1 if drop in ("first", "if_binary") else 0
                                for cat in c_cats[dropped:]:
                                    new_cols.append(f"{c_name}={cat}")
                            current = new_cols
                            continue
                        except Exception:
                            pass

                    if isinstance(step_obj, VarianceThreshold) and hasattr(step_obj, "get_support"):
                        try:
                            mask = step_obj.get_support()
                            if mask is not None and len(mask) == len(current):
                                current = [n for n, keep in zip(current, mask) if keep]
                                continue
                        except Exception:
                            pass

                    try:
                        probe = pd.DataFrame(np.zeros((5, len(current))), columns=current)
                        out = step_obj.transform(probe)
                        if hasattr(out, "columns"):
                            current = list(out.columns)
                            continue
                    except Exception:
                        pass

                names.extend(current)
                continue

            if hasattr(trans, "get_feature_names_out"):
                try:
                    sub = list(trans.get_feature_names_out(cols_list))
                    names.extend([str(s) for s in sub])
                    continue
                except Exception:
                    pass

            names.extend([str(c) for c in cols_list])

        return names

    try:
        names = reconstruct_ct_feature_names(ct)
    except Exception:
        try:
            Xt_ct = ct.transform(X)
            if hasattr(Xt_ct, "columns"):
                names = list(Xt_ct.columns)
            else:
                names = [f"feature_{i}" for i in range(Xt_ct.shape[1])]
        except Exception:
            names = list(X.columns)

    steps_after = preproc.steps[ct_idx+1:] if ct_idx is not None else preproc.steps

    def try_df_cols_after_step(step, colnames):
        try:
            probe = pd.DataFrame(np.zeros((5, len(colnames))), columns=colnames)
            out = step.transform(probe)
            if hasattr(out, "columns"):
                return list(out.columns)
        except Exception:
            return None

    for __, step_obj in steps_after:
        cols = try_df_cols_after_step(step_obj, names)
        if cols:
            names = cols
            continue

        if isinstance(step_obj, VarianceThreshold) and hasattr(step_obj, "get_support"):
            try:
                mask = step_obj.get_support()
                if mask is not None and len(mask) == len(names):
                    names = [n for n, keep in zip(names, mask) if keep]
                    continue
            except Exception:
                pass

        if hasattr(step_obj, "selected_features_") and getattr(step_obj, "selected_features_", None):
            sel = [n for n in step_obj.selected_features_ if n in set(names)]
            if sel:
                names = sel
                continue

        if step_obj.__class__.__name__ == "AddLogTimeInteractions":
            target_feats = getattr(step_obj, "columns_", None) or getattr(step_obj, "feature_names", None)
            if not target_feats:
                target_feats = names
            inter = [f"{c}__x__logt" for c in target_feats if c in names]
            names = names + inter
            continue

    return [str(n) for n in names]


def generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=int)

    # Categorical risk & tariff
    risk_levels = rng.choice(["low", "med", "high"], size=n, p=[0.4, 0.4, 0.2])
    tariffs = rng.choice(["A", "B", "C", "D"], size=n, p=[0.3, 0.4, 0.2, 0.1])

    # Numeric covariates
    debit = rng.lognormal(mean=6.0, sigma=0.5, size=n)
    credit = rng.lognormal(mean=6.2, sigma=0.55, size=n)
    balance = rng.normal(loc=500.0, scale=200.0, size=n)
    past_due = rng.lognormal(mean=4.0, sigma=0.8, size=n)
    total_set = rng.poisson(lam=2.0, size=n).astype(float)
    active_set = np.maximum(0.0, total_set - rng.poisson(lam=1.0, size=n))
    defaulted_set = np.maximum(0.0, rng.binomial(n=3, p=0.1, size=n)).astype(float)

    # True hazard (simulate PH with some effects)
    base_hazard = 0.02  # per month
    risk_map = {"low": -0.3, "med": 0.0, "high": 0.4}
    tariff_map = {"A": 0.0, "B": 0.1, "C": 0.2, "D": -0.1}
    linpred = (
        0.0005 * (debit - debit.mean())
        + 0.0002 * (credit - credit.mean())
        + 0.0010 * (past_due - past_due.mean())
        + 0.0008 * (balance - balance.mean())
        + 0.05 * (total_set - total_set.mean())
        + 0.03 * (active_set - active_set.mean())
        + 0.06 * (defaulted_set - defaulted_set.mean())
        + np.vectorize(risk_map.get)(risk_levels)
        + np.vectorize(tariff_map.get)(tariffs)
    )
    # time-to-event ~ exponential with rate = base * exp(linpred)
    rates = base_hazard * np.exp(linpred)
    durations = rng.exponential(1.0 / np.maximum(rates, 1e-6))
    durations = np.clip(durations, 0.1, 72.0)  # cap at 6 years

    # Censoring ~ about 60%
    censoring_time = rng.exponential(24.0, size=n)  # expected censor around 24 months
    event = (durations <= censoring_time).astype(int)
    observed_time = np.minimum(durations, censoring_time)

    df = pd.DataFrame(
        {
            ID_COL: ids,
            DURATION_COL: observed_time.astype(float),
            EVENT_COL: event.astype(int),
            "risk_level_coarse_col": risk_levels,
            "debit_exp_smooth_col": debit,
            "credit_exp_smooth_col": credit,
            "balance_exp_smooth_col": balance,
            "past_due_balance_exp_smooth_col": past_due,
            "total_settlements_col": total_set,
            "active_settlements_col": active_set,
            "defaulted_settlements_col": defaulted_set,
            "tariff_coarse_col": tariffs,
        }
    )

    # Inject some missingness
    for col in NUMERIC_COLS:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan
    for col in CATEGORICAL_COLS:
        mask = rng.random(n) < 0.02
        df.loc[mask, col] = np.nan

    return df


# ========================================
# Refactored main workflow functions
# ========================================

@time_function
def setup_environment(args) -> None:
    """Initialize logging, random seeds, and output directory."""
    ensure_dir(args.output_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "run.log"), mode="w", encoding="utf-8"),
        ],
    )
    set_global_seed(args.random_seed)
    logging.info("Starting run with seed=%d", args.random_seed)


@time_function
def filter_thin_strata(X_raw: pd.DataFrame, y: np.ndarray, strata_cols: List[str], 
                      min_obs_per_stratum: int = 10, min_events_per_stratum: int = 2) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """
    Filter out observations from strata that are too thin for reliable modeling.
    
    Parameters
    ----------
    X_raw : pd.DataFrame
        Raw feature data
    y : np.ndarray
        Survival data (structured array)
    strata_cols : List[str]
        Columns used for stratification
    min_obs_per_stratum : int
        Minimum number of observations required per stratum
    min_events_per_stratum : int
        Minimum number of events required per stratum
        
    Returns
    -------
    X_filtered : pd.DataFrame
        Filtered feature data
    y_filtered : np.ndarray
        Filtered survival data
    filter_info : Dict[str, Any]
        Information about filtering process
    """
    if not strata_cols:
        return X_raw, y, {"thin_strata_removed": [], "observations_dropped": 0}
    
    # Create combined strata key for each observation
    strata_keys = []
    for idx in range(len(X_raw)):
        key_parts = []
        for col in strata_cols:
            if col in X_raw.columns:
                val = str(X_raw.iloc[idx][col])
            else:
                val = "Missing"
            key_parts.append(f"{col}={val}")
        strata_keys.append("__".join(key_parts))
    
    # Count observations and events per stratum
    strata_df = pd.DataFrame({
        'strata_key': strata_keys,
        'event': y["event"],
        'time': y["time"]
    })
    
    strata_stats = strata_df.groupby('strata_key').agg({
        'event': ['count', 'sum'],
        'time': 'count'
    }).round(2)
    
    # Flatten column names
    strata_stats.columns = ['obs_count', 'event_count', 'time_count']
    
    # Identify valid strata
    valid_strata = strata_stats[
        (strata_stats['obs_count'] >= min_obs_per_stratum) & 
        (strata_stats['event_count'] >= min_events_per_stratum)
    ].index.tolist()
    
    # Identify thin strata that will be removed
    thin_strata = strata_stats[
        (strata_stats['obs_count'] < min_obs_per_stratum) | 
        (strata_stats['event_count'] < min_events_per_stratum)
    ]
    
    if len(thin_strata) > 0:
        logging.info("Removing thin strata with insufficient observations/events:")
        for stratum, stats in thin_strata.iterrows():
            logging.info(f"  {stratum}: {int(stats['obs_count'])} obs, {int(stats['event_count'])} events")
    
    # Filter data to keep only valid strata
    strata_series = pd.Series(strata_keys)
    valid_mask = strata_series.isin(valid_strata)
    
    X_filtered = X_raw[valid_mask].reset_index(drop=True)
    y_filtered = y[valid_mask]
    
    observations_dropped = len(X_raw) - len(X_filtered)
    
    filter_info = {
        "thin_strata_removed": thin_strata.index.tolist(),
        "observations_dropped": observations_dropped,
        "strata_stats_before": strata_stats.to_dict('index'),
        "valid_strata_count": len(valid_strata),
        "thin_strata_count": len(thin_strata)
    }
    
    if observations_dropped > 0:
        logging.info(f"Filtered out {observations_dropped} observations from {len(thin_strata)} thin strata")
        logging.info(f"Remaining: {len(X_filtered)} observations across {len(valid_strata)} strata")
    else:
        logging.info("No thin strata found - all strata have sufficient observations")
    
    return X_filtered, y_filtered, filter_info


@time_function
def load_and_prepare_data(data_path: Optional[str], strata_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """Load, validate, clean data and prepare X, y structures for modeling."""
    df_raw = read_data(data_path)
    validate_schema(df_raw)
    df = clean_data(df_raw).reset_index(drop=True)
    
    # Log dataset statistics
    n_rows = len(df)
    n_events = int(df[EVENT_COL].sum())
    cens_rate = censoring_rate(df[EVENT_COL].to_numpy())
    logging.info("Rows: %d | Events: %d | Censoring rate: %.2f", n_rows, n_events, cens_rate)

    if n_events == 0 or n_events == n_rows:
        raise ValueError("All-censored or all-event data detected. Model cannot be trained.")

    # Build structured arrays for survival analysis
    y = build_y_structured(df)
    X_raw = df.drop(columns=[ID_COL, DURATION_COL, EVENT_COL])
    
    # Filter thin strata if stratification columns are provided
    filter_info = {"thin_strata_removed": [], "observations_dropped": 0}
    if strata_cols:
        X_raw, y, filter_info = filter_thin_strata(X_raw, y, strata_cols)
        
        # Update df to match filtered data
        if filter_info["observations_dropped"] > 0:
            # Rebuild df from filtered data
            df_filtered = pd.concat([
                df[[ID_COL, DURATION_COL, EVENT_COL]].iloc[:len(X_raw)],
                X_raw
            ], axis=1)
            df = df_filtered.reset_index(drop=True)
    
    return df, y, X_raw, filter_info


@time_function
def run_model_selection(X_raw: pd.DataFrame, y: np.ndarray, seed: int, strata_cols: List[str] = None) -> Tuple[List[CVResult], CVResult]:
    """Execute grid search CV for candidate models and select best via tiebreaking."""
    k_values = [10, 20, 40, 80, "all"]
    
    # Use provided strata_cols or default
    if strata_cols is None:
        strata_cols = [c.tariff_coarse_col, c.risk_level_coarse_col]
    
    # Test if stratification is needed
    needs_stratification, strata_tests = test_stratification_need(X_raw, y, strata_cols)
    
    # Build original model candidates
    model_A, grid_A = build_model_A(k_values=k_values)
    model_B, grid_B = build_model_B()
    model_C, grid_C = build_model_C(k_values=k_values)

    candidates: List[CVResult] = []
    
    # Always run original Models A, B, C
    logging.info("Running GridSearch for Model A (CoxPH + univariate)...")
    resA = run_grid_search("ModelA_CoxPH_UV", model_A, grid_A, X_raw, y, seed=seed)
    candidates.append(resA)
    logging.info("Model A best C-index: %.4f with params: %s", resA.best_score_, json.dumps(resA.best_params_))

    logging.info("Running GridSearch for Model B (Coxnet)...")
    resB = run_grid_search("ModelB_Coxnet", model_B, grid_B, X_raw, y, seed=seed)
    candidates.append(resB)
    logging.info("Model B best C-index: %.4f with params: %s", resB.best_score_, json.dumps(resB.best_params_))

    logging.info("Running GridSearch for Model C (CoxPH + log(time) interactions + univariate)...")
    resC = run_grid_search("ModelC_CoxPH_LogT", model_C, grid_C, X_raw, y, seed=seed)
    candidates.append(resC)
    logging.info("Model C best C-index: %.4f with params: %s", resC.best_score_, json.dumps(resC.best_params_))

    # Run Model D if stratification is statistically justified
    if needs_stratification:
        logging.info("Stratification is statistically justified. Running Model D...")
        model_D, grid_D = build_model_D_with_strata(k_values=[5, 10, 20], strata_cols=strata_cols)
        resD = run_grid_search("ModelD_CoxPH_Strata", model_D, grid_D, X_raw, y, seed=seed)
        candidates.append(resD)
        logging.info("Model D best C-index: %.4f with params: %s", resD.best_score_, json.dumps(resD.best_params_))
        
        # Build and test stratified versions of all models
        logging.info("Building stratified model variants...")
        stratified_models = build_stratified_models_with_strata(k_values, strata_cols)
        
        for model_name, (strat_pipe, strat_grid) in stratified_models.items():
            try:
                logging.info(f"Running GridSearch for {model_name}...")
                res_strat = run_grid_search(model_name, strat_pipe, strat_grid, X_raw, y, seed=seed)
                candidates.append(res_strat)
                logging.info(f"{model_name} best C-index: %.4f with params: %s", 
                           res_strat.best_score_, json.dumps(res_strat.best_params_))
            except Exception as e:
                logging.warning(f"Failed to run {model_name}: {e}")
                continue
    else:
        logging.info("Stratification not statistically justified. Skipping stratified model variants.")

    # Select best model using tiebreaking
    times = np.array(PREDICTION_MONTHS, dtype=float)
    best = tiebreak_with_ibs(candidates, X_raw, y, times=times, tol=1e-6)
    logging.info("Selected model: %s | C-index=%.4f | IBS (if evaluated)=%s",
                 best.name, best.best_score_, f"{best.ibs_tiebreak_:.4f}" if best.ibs_tiebreak_ is not None else "n/a")
    
    return candidates, best

def build_model_D_with_strata(k_values: List[Any], strata_cols: List[str]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """
    Model D: Stratified Cox model simulation via interaction terms
    
    Simulates stratification by creating interaction terms between all features 
    and the specified stratification variables.
    This allows different baseline hazards and feature effects for each stratum.
    """
    # Apply strata interactions BEFORE preprocessing
    pipe = Pipeline(
        steps=[
            ("strata", AddStrataInteractions(strata_cols=strata_cols)),
            ("pre", build_preprocessor()),
            ("uv", UnivariateCoxSelector(k="all")),
            ("coxph", CoxPHSurvivalAnalysis()),
        ]
    )
    param_grid = {"uv__k": k_values}
    return pipe, param_grid


def build_stratified_models_with_strata(k_values: List[Any], strata_cols: List[str]) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """
    Build stratified versions of all model types with custom stratification columns.
    Returns dictionary mapping model names to (pipeline, param_grid) tuples.
    """
    
    # Create decorator instance with custom strata_cols
    stratified_trainer_custom = StratifiedModelTrainer(strata_cols=strata_cols)
    
    @stratified_trainer_custom
    def stratified_model_A(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_A(k_values)
    
    @stratified_trainer_custom  
    def stratified_model_B() -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_B()
    
    @stratified_trainer_custom
    def stratified_model_C(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_C(k_values)
    
    @stratified_trainer_custom
    def stratified_model_D(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
        return build_model_D_with_strata(k_values, strata_cols)
    
    # Build all stratified models
    stratified_models = {}
    
    model_A_strat, grid_A_strat = stratified_model_A(k_values=k_values)
    stratified_models["ModelA_Stratified"] = (model_A_strat, grid_A_strat)
    
    model_B_strat, grid_B_strat = stratified_model_B()
    stratified_models["ModelB_Stratified"] = (model_B_strat, grid_B_strat)
    
    model_C_strat, grid_C_strat = stratified_model_C(k_values=k_values)
    stratified_models["ModelC_Stratified"] = (model_C_strat, grid_C_strat)
    
    model_D_strat, grid_D_strat = stratified_model_D(k_values=[5, 10, 20])
    stratified_models["ModelD_Stratified"] = (model_D_strat, grid_D_strat)
    
    return stratified_models



@time_function
def generate_diagnostic_plots(final_model: Pipeline, X_raw: pd.DataFrame, y: np.ndarray, 
                            output_dir: str) -> None:
    """Generate Brier curve and calibration plots for model validation."""
    times = np.array(PREDICTION_MONTHS, dtype=float)
    
    # Brier curve analysis
    try:
        surv_funcs_full = final_model.predict_survival_function(X_raw)
        preds_full = np.vstack([sf(times) for sf in surv_funcs_full])
        _, brier = brier_score(survival_train=y, survival_test=y, estimate=preds_full, times=times)
        ibs_full = integrated_brier_score(survival_train=y, survival_test=y, estimate=preds_full, times=times)
        logging.info("Final model IBS (1-36m): %.4f", float(ibs_full))

        # Plot Brier curve
        brier_path = os.path.join(output_dir, "brier_curve_1_36m.png")
        plot_brier_curve(times, brier, brier_path, title="Brier Score (1-36 months)")
        logging.info("Saved Brier curve: %s", brier_path)
    except Exception as e:
        logging.warning("Could not compute/plot Brier diagnostics: %r", e)

    # Calibration plots at key time points
    for t in [12.0, 24.0, 36.0]:
        try:
            outp = os.path.join(output_dir, f"calibration_{int(t)}m.png")
            calibration_plot_at_time(
                final_model, X_raw, y, t=t, out_path=outp, title=f"Calibration at {int(t)} months"
            )
            logging.info("Saved calibration plot at %dm: %s", int(t), outp)
        except Exception as e:
            logging.warning("Calibration plot at %dm failed: %r", int(t), e)


@time_function
def save_model_artifacts(final_model: Pipeline, candidates: List[CVResult], best: CVResult,
                        X_raw: pd.DataFrame, y: np.ndarray, output_dir: str, strata_cols: List[str] = None,
                        filter_info: Dict[str, Any] = None) -> None:
    """Save model, CV results, feature importance, and performance metrics."""
    # Save final fitted model
    model_path = os.path.join(output_dir, "final_model.joblib")
    dump(final_model, model_path)
    logging.info("Saved final model: %s", model_path)

    # Save CV results for each candidate
    for res in candidates:
        if res.cv_results_ is not None:
            path = os.path.join(output_dir, f"cv_results_{res.name}.csv")
            res.cv_results_.to_csv(path, index=False)
            logging.info("Saved CV results: %s", path)

    # Export feature importance/coefficients
    _export_feature_importance(final_model, X_raw, y, output_dir)
    
    # Export model performance metrics
    _export_model_metrics(candidates, best, final_model, X_raw, y, output_dir, strata_cols, filter_info)


def _export_feature_importance(final_model: Pipeline, X_raw: pd.DataFrame, y: np.ndarray, 
                              output_dir: str) -> None:
    """Export feature coefficients and importance measures."""
    try:
        from scipy import stats
        from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

        # Locate terminal estimator
        last_step_name, last_est = final_model.steps[-1]
        feat_path = os.path.join(output_dir, "feature_importance.csv")
        full_sum_path = os.path.join(output_dir, "cox_full_summary.csv")

        # Get meaningful feature names for the FINAL pipeline (selected features)
        feature_names = get_feature_names_from_pipeline(final_model, X_raw, y)

        # Feature importance for final selected model
        if isinstance(last_est, CoxPHSurvivalAnalysis):
            coefs = getattr(last_est, "coef_", None)
            if coefs is not None:
                coefs = np.asarray(coefs).ravel()
                hr = np.exp(coefs)

                # Ensure we have enough feature names
                if len(feature_names) < len(coefs):
                    feature_names.extend(
                        [f"feature_{i}" for i in range(len(feature_names), len(coefs))]
                    )

                # Compute dataset-average survival at horizons & per-feature +1 effect
                horizons = [12.0, 24.0, 36.0]
                surv_baseline = {h: np.nan for h in horizons}
                surv_matrix = {h: np.array([]) for h in horizons}
                try:
                    sfs = final_model.predict_survival_function(X_raw, return_array=False)
                    for h in horizons:
                        vals = np.array([sf([h])[0] for sf in sfs], dtype=float)
                        surv_matrix[h] = vals
                        if vals.size > 0:
                            surv_baseline[h] = float(np.mean(vals))
                except Exception as _e:
                    logging.warning("Could not compute survival functions: %r", _e)

                df_coef = pd.DataFrame({
                    "feature": feature_names[:len(coefs)],
                    "coef": coefs,
                    "hazard_ratio": hr,
                })

                # Baseline survival columns (same value on all rows for convenience)
                for h in horizons:
                    df_coef[f"surv_{int(h)}m_baseline"] = surv_baseline[h]

                # Per-feature +1 effect: mean_i S_i(t)**HR_j
                for h in horizons:
                    base_vec = surv_matrix[h]
                    if base_vec.size == 0:
                        df_coef[f"surv_{int(h)}m_effect(+1)"] = np.nan
                    else:
                        df_coef[f"surv_{int(h)}m_effect(+1)"] = [
                            float(np.mean(base_vec ** hr_j)) for hr_j in hr
                        ]

                df_coef = df_coef.sort_values("hazard_ratio", ascending=False)
                df_coef.to_csv(feat_path, index=False)
                logging.info("Saved CoxPH coefficients with survival columns: %s", feat_path)

        elif isinstance(last_est, CoxnetSurvivalAnalysis):
            # Keep existing Coxnet export (no survival columns since baseline is different)
            coef = getattr(last_est, "coef_", None)
            alphas = getattr(last_est, "alphas_", None)
            if coef is not None and alphas is not None:
                coef_vec = np.asarray(coef)[:, -1]
                hr = np.exp(coef_vec)
                if len(feature_names) < len(coef_vec):
                    feature_names.extend(
                        [f"feature_{i}" for i in range(len(feature_names), len(coef_vec))]
                    )
                df_coef = pd.DataFrame(
                    {
                        "feature": feature_names[:len(coef_vec)],
                        "coef": coef_vec,
                        "hazard_ratio": hr,
                    }
                ).sort_values("hazard_ratio", ascending=False)
                df_coef.to_csv(feat_path, index=False)
                logging.info("Saved Coxnet coefficients: %s", feat_path)

        # Export full summary with p-values (skip for simplicity - would need lifelines refit)
        # This part would require refitting with lifelines for p-values as discussed earlier
        
    except Exception as e:
        logging.warning("Could not export feature importance: %r", e)


def _export_model_metrics(candidates: List[CVResult], best: CVResult, final_model: Pipeline,
                         X_raw: pd.DataFrame, y: np.ndarray, output_dir: str, strata_cols: List[str] = None,
                         filter_info: Dict[str, Any] = None) -> None:
    """Export comprehensive model performance metrics and comparison."""
    try:
        times = np.array(PREDICTION_MONTHS, dtype=float)
        
        # Use provided strata_cols or default
        if strata_cols is None:
            strata_cols = [c.tariff_coarse_col, c.risk_level_coarse_col]
        
        # Test stratification and save results
        needs_stratification, strata_tests = test_stratification_need(X_raw, y, strata_cols)
        
        # Save stratification test results
        strata_df = pd.DataFrame([
            {"variable": col, "p_value": p_val, "significant": p_val < 0.05}
            for col, p_val in strata_tests.items()
        ])
        strata_path = os.path.join(output_dir, "stratification_tests.csv")
        strata_df.to_csv(strata_path, index=False)
        logging.info("Saved stratification test results: %s", strata_path)
        
        # Save data filtering information if available
        if filter_info and filter_info.get("observations_dropped", 0) > 0:
            filter_df = pd.DataFrame([
                {"metric": "thin_strata_removed", "value": len(filter_info["thin_strata_removed"]), 
                 "description": "Number of thin strata removed during preprocessing"},
                {"metric": "observations_dropped", "value": filter_info["observations_dropped"], 
                 "description": "Number of observations dropped due to thin strata"},
                {"metric": "valid_strata_count", "value": filter_info["valid_strata_count"], 
                 "description": "Number of strata retained for modeling"},
            ])
            filter_path = os.path.join(output_dir, "data_filtering_summary.csv")
            filter_df.to_csv(filter_path, index=False)
            logging.info("Saved data filtering summary: %s", filter_path)
        
        # Existing metrics export code...
        final_ibs = best.ibs_tiebreak_
        if final_ibs is None:
            try:
                final_ibs = compute_ibs(final_model, X_raw, y, times)
            except Exception:
                final_ibs = None

        # Create metrics summary
        metrics_data = {
            "metric": ["C-index_CV", "IBS_1-36m", "Stratification_needed", "Strata_columns_tested"],
            "value": [best.best_score_, final_ibs, needs_stratification, ", ".join(strata_cols)],
            "description": [
                "Cross-validated Harrell's C-index (higher is better)",
                "Integrated Brier Score 1-36 months (lower is better)",
                "Whether stratification is statistically justified",
                "Columns tested for stratification"
            ]
        }
        
        # Add filtering metrics if available
        if filter_info and filter_info.get("observations_dropped", 0) > 0:
            metrics_data["metric"].extend(["Observations_dropped", "Thin_strata_removed"])
            metrics_data["value"].extend([filter_info["observations_dropped"], len(filter_info["thin_strata_removed"])])
            metrics_data["description"].extend([
                "Observations dropped due to thin strata filtering",
                "Number of thin strata removed during preprocessing"
            ])
        
        df_metrics = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(output_dir, "model_metrics.csv")
        df_metrics.to_csv(metrics_path, index=False)
        logging.info("Saved model metrics: %s", metrics_path)
        
        # Detailed model comparison
        detailed_metrics = []
        for candidate in candidates:
            detailed_metrics.append({
                "model": candidate.name,
                "c_index_cv": candidate.best_score_,
                "ibs_1_36m": candidate.ibs_tiebreak_,
                "best_params": json.dumps(candidate.best_params_),
                "selected": candidate.name == best.name
            })
        
        df_detailed = pd.DataFrame(detailed_metrics)
        detailed_path = os.path.join(output_dir, "model_comparison.csv")
        df_detailed.to_csv(detailed_path, index=False)
        logging.info("Saved model comparison: %s", detailed_path)
        
    except Exception as e:
        logging.warning("Could not export model metrics: %r", e)


@time_function
def print_final_summary(best: CVResult, final_model: Pipeline, X_raw: pd.DataFrame, 
                       y: np.ndarray) -> None:
    """Print concise summary of final model performance and top features."""
    try:
        times = np.array(PREDICTION_MONTHS, dtype=float)
        last_est = final_model.steps[-1][1]
        feature_names = get_feature_names_from_pipeline(final_model, X_raw, y)
        
        # Compute final IBS for display if not available
        display_ibs = best.ibs_tiebreak_
        if display_ibs is None:
            try:
                display_ibs = compute_ibs(final_model, X_raw, y, times)
            except Exception:
                display_ibs = None
        
        print("\n=== Final Model Summary ===")
        print(f"Chosen pipeline: {best.name}")
        print(f"Best hyperparameters: {json.dumps(best.best_params_)}")
        print(f"CV C-index (mean): {best.best_score_:.4f}")
        
        if display_ibs is not None:
            print(f"Integrated Brier Score (1-36m): {display_ibs:.4f}")
        else:
            print("Integrated Brier Score (1-36m): Could not compute")
            
        if best.ibs_tiebreak_ is not None:
            print(f"IBS tiebreak value: {best.ibs_tiebreak_:.4f}")

        # Display top coefficients by magnitude
        if isinstance(last_est, CoxPHSurvivalAnalysis):
            coefs = getattr(last_est, "coef_", None)
            if coefs is not None:
                coefs = np.asarray(coefs).ravel()
                top_idx = np.argsort(np.abs(coefs))[-10:][::-1]
                print("Top CoxPH coefficients (|coef|):")
                for i in top_idx:
                    fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    print(f"  {fname}: coef={coefs[i]: .4f}, HR={math.exp(coefs[i]): .4f}")
                    
        elif isinstance(last_est, CoxnetSurvivalAnalysis):
            coef = getattr(last_est, "coef_", None)
            if coef is not None:
                vec = np.asarray(coef)[:, -1]
                nz = np.where(np.abs(vec) > 1e-8)[0]
                top_idx = nz[np.argsort(np.abs(vec[nz]))[-10:][::-1]]
                print("Top Coxnet coefficients (|coef|):")
                for i in top_idx:
                    fname = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    print(f"  {fname}: coef={vec[i]: .4f}, HR={math.exp(vec[i]): .4f}")
    except Exception as e:
        logging.warning("Failed to print model summary: %r", e)



@time_function
def main(
    data_path: Optional[str] = None,
    output_dir: str = "./artifacts",
    random_seed: int = 42,
    config_yaml: Optional[str] = None,
    strata_cols: Optional[List[str]] = None,
    use_cli: bool = True,
):
    """
    Orchestrate the complete Cox survival modeling pipeline.
    
    Parameters
    ----------
    data_path : Optional[str]
        Path to input data file (CSV, Parquet, or Pickle)
    output_dir : str
        Directory to save artifacts
    random_seed : int
        Random seed for reproducibility
    config_yaml : Optional[str]
        Path to YAML configuration file (not implemented)
    strata_cols : Optional[List[str]]
        Column names to test for stratification. If None, defaults to 
        ['typeoftariff_coarse', 'risk_level_coarse']
    use_cli : bool
        Whether to parse command line arguments
    
    Design rationale:
    - Each major operation is delegated to a focused function with single responsibility
    - Timing decorator provides consistent execution time measurement across all operations
    - Clean separation between data prep, model selection, validation, and artifact generation
    - Minimal parameter passing - functions return what subsequent steps need
    - Error handling contained within each function to prevent cascade failures
    - Supports both CLI and direct Python invocation with parameters.
    """
    
    if use_cli:
        parser = argparse.ArgumentParser(description="Train Cox survival model and output 1..36 month survival probabilities.")
        parser.add_argument("--data-path", type=str, default=None, help="Path to CSV or Parquet input.")
        parser.add_argument("--output-dir", type=str, default="./artifacts", help="Directory to save artifacts.")
        parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
        parser.add_argument("--config-yaml", type=str, default=None, help="Optional YAML config to override defaults.")
        parser.add_argument("--strata-cols", type=str, nargs='+', default=None, 
                          help="Column names to test for stratification (space-separated).")
        args = parser.parse_args()
        data_path = args.data_path
        output_dir = args.output_dir
        random_seed = args.random_seed
        config_yaml = args.config_yaml
        strata_cols = args.strata_cols
    else:
        # Use parameters passed directly to the function
        class Args:
            pass
        args = Args()
        args.data_path = data_path
        args.output_dir = output_dir
        args.random_seed = random_seed
        args.config_yaml = config_yaml
        args.strata_cols = strata_cols

    # Set default strata_cols if not provided
    if args.strata_cols is None:
        args.strata_cols = [c.tariff_coarse_col, c.risk_level_coarse_col]
    
    # Step 1: Environment setup (logging, seeds, directories)
    setup_environment(args)

    logging.info(f"Using stratification columns: {args.strata_cols}")

    # Step 2: Data loading and preparation (now includes thin strata filtering)
    df, y, X_raw, filter_info = load_and_prepare_data(args.data_path, args.strata_cols)

    # Step 3: Model selection via cross-validation
    candidates, best_model = run_model_selection(X_raw, y, args.random_seed, args.strata_cols)

    # Step 4: Refit final model and generate diagnostics
    final_model = best_model.best_estimator_
    if final_model is None:
        raise RuntimeError("No best_estimator_ found.")

    final_model_refit = final_model.fit(X_raw, y)
    generate_diagnostic_plots(final_model_refit, X_raw, y, args.output_dir)

    # Step 5: Save all artifacts (model, results, metrics)
    save_model_artifacts(final_model_refit, candidates, best_model, X_raw, y, args.output_dir, args.strata_cols, filter_info)

    # Step 6: Print final summary
    print_final_summary(best_model, final_model_refit, X_raw, y)

    # Final log message
    logging.info("Artifacts saved under: %s", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    # Suppress some benign warnings for cleaner logs
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    use_cli=False
    random_seed=42
    config_yaml=None
    strata_cols=['typeoftariff_coarse']

    data_path='input_data/sample/survival_inputs_sample2000.csv'
    output_dir="./output_data/sample"

    # data_path='input_data/complete/account_snapshot_kpis.csv'
    # output_dir="./output_data/complete"

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = os.path.join(output_dir, script_name)

    main(use_cli=False, data_path=data_path, output_dir=output_dir, random_seed=random_seed, config_yaml=config_yaml, strata_cols=strata_cols)