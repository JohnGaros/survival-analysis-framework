# LAPACK Error Root Cause Analysis

**Date:** 2025-10-23
**Error:** `ValueError: LAPACK reported an illegal value in 5-th argument`
**Context:** Occurs when running `src/main.py` with survival_inputs_sample2000.csv
**Status:** ✅ **RESOLVED** - All Priority 1 fixes implemented successfully

## Executive Summary

The LAPACK error is caused by **numerical instability in the Cox Proportional Hazards model** due to:

1. **No missing value handling** - Missing values passed directly to CoxPH
2. **No feature variance filtering** - Near-zero variance features cause singular matrices
3. **No regularization** - Unregularized CoxPH sensitive to multicollinearity
4. **No feature selection** - Too many correlated features relative to events
5. **Missing preprocessing steps** - No imputation indicators or rare category handling

The working script (`cox_survival_v9.py`) avoids this error through comprehensive preprocessing and regularization.

---

## Side-by-Side Comparison

### Current Framework (BROKEN)

**File:** `src/survival_framework/data.py`

```python
def make_preprocessor(numeric=None, categorical=None) -> ColumnTransformer:
    """Simple preprocessing with NO missing value handling."""
    numeric = numeric if numeric is not None else NUM_COLS
    categorical = categorical if categorical is not None else CAT_COLS

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),  # No imputation!
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return pre
```

**Issues:**
- ❌ Missing values → NaN in StandardScaler → NaN in CoxPH → LAPACK error
- ❌ No variance threshold filtering
- ❌ No rare category grouping
- ❌ No missing indicators

**File:** `src/survival_framework/models.py`

```python
@dataclass
class CoxPHWrapper(BaseSurvivalModel):
    """CoxPH with NO regularization."""
    name: str = "cox_ph"
    model: CoxPHSurvivalAnalysis = None

    def __post_init__(self):
        if self.model is None:
            self.model = CoxPHSurvivalAnalysis(n_iter=200, tol=1e-9)
            # ❌ No alpha parameter (regularization)
            # ❌ No feature selection
            # ❌ Direct fitting on all features
```

---

### Working Script (cox_survival_v9.py)

**Preprocessing Pipeline:**

```python
def build_preprocessor(rare_threshold: float = 0.01) -> Pipeline:
    """Comprehensive preprocessing with imputation and variance filtering."""

    # Categorical branch
    cat_pipe = Pipeline(
        steps=[
            ("rare", RareCategoryGrouper(threshold=rare_threshold)),  # ✅ Group rare levels
            ("impute", ImputeWithIndicator(strategy="most_frequent")),  # ✅ Impute + indicators
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    # Numeric branch
    num_pipe = Pipeline(
        steps=[
            ("impute", ImputeWithIndicator(strategy="median")),  # ✅ Impute + indicators
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
        verbose_feature_names_out=False,
    )

    # ✅ Drop near-zero variance features (prevents singular matrices)
    full = Pipeline(
        steps=[
            ("pre", pre),
            ("varth", VarianceThreshold(threshold=1e-12)),
        ]
    )

    return full
```

**Model A (Basic CoxPH with Feature Selection):**

```python
def build_model_A(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """Preprocessing -> UnivariateCoxSelector -> CoxPH"""
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("uv", UnivariateCoxSelector(k="all")),  # ✅ Feature selection
            ("coxph", CoxPHSurvivalAnalysis()),
        ]
    )
    param_grid = {"uv__k": k_values}
    return pipe, param_grid
```

**Model C (With Regularization):**

```python
def build_model_C(k_values: List[Any]) -> Tuple[Pipeline, Dict[str, List[Any]]]:
    """Preprocessing -> Log Time Interactions -> Feature Selection -> Regularized CoxPH"""
    pre = build_preprocessor()
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("logt", AddLogTimeInteractions()),
            ("uv", UnivariateCoxSelector(k="all")),
            ("coxph", CoxPHSurvivalAnalysis(alpha=1)),  # ✅ L2 regularization
        ]
    )
    param_grid = {"uv__k": [5, 10, 20]}  # ✅ Fewer features
    return pipe, param_grid
```

---

## Root Cause Breakdown

### 1. Missing Value Handling

**Problem:**
```python
# Current Framework
StandardScaler().fit_transform(X)  # If X has NaN → output has NaN
CoxPHSurvivalAnalysis().fit(X_scaled, y)  # NaN → LAPACK error
```

**Solution from cox_survival_v9.py:**
```python
class ImputeWithIndicator(BaseEstimator, TransformerMixin):
    """Impute missing values AND add indicator columns for missingness."""

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        # add_indicator=True creates binary columns for missing values
        self.imputer_ = SimpleImputer(
            strategy=self.strategy,
            fill_value=self.fill_value,
            add_indicator=True
        )
        self.imputer_.fit(X)
        return self
```

**Why this works:**
- Imputes missing values (no NaN in CoxPH)
- Adds binary indicators to preserve missingness information
- Prevents LAPACK from encountering NaN values

---

### 2. Variance Threshold Filtering

**Problem:**
```python
# Current Framework: No variance filtering
# Features with zero or near-zero variance → singular covariance matrix → LAPACK error
```

**Solution from cox_survival_v9.py:**
```python
Pipeline([
    ("pre", preprocessor),
    ("varth", VarianceThreshold(threshold=1e-12)),  # Remove constant features
])
```

**Why this works:**
- Removes features with variance < 1e-12
- Prevents singular matrices in CoxPH optimization
- Reduces multicollinearity

---

### 3. Regularization

**Problem:**
```python
# Current Framework
CoxPHSurvivalAnalysis(n_iter=200, tol=1e-9)  # alpha=0 (default, no regularization)
```

**Solution from cox_survival_v9.py:**
```python
CoxPHSurvivalAnalysis(alpha=1)  # L2 regularization (Ridge penalty)
```

**Why this works:**
- Regularization adds `alpha * ||beta||^2` to the objective
- Stabilizes coefficient estimates when features are correlated
- Prevents coefficient explosion in ill-conditioned matrices
- Default `alpha=0` means no regularization → unstable

---

### 4. Feature Selection

**Problem:**
```python
# Current Framework: Uses ALL features directly
# With 2000 samples and potential 50+ features after one-hot encoding:
# - High dimensionality relative to events
# - Many correlated features
# - Multicollinearity issues
```

**Solution from cox_survival_v9.py:**
```python
class UnivariateCoxSelector(BaseEstimator, TransformerMixin):
    """Univariate feature screening using lifelines CoxPHFitter.

    Ranks features by:
    1. Harrell's C-index (univariate Cox regression)
    2. P-value < threshold

    Selects top k features.
    """

    def __init__(self, k="all", pval_threshold=0.05, min_features=5):
        self.k = k
        self.pval_threshold = pval_threshold
        self.min_features = min_features
```

**Why this works:**
- Reduces dimensionality before CoxPH
- Removes statistically insignificant features
- Reduces multicollinearity
- Improves numerical stability

---

### 5. Rare Category Handling

**Problem:**
```python
# Current Framework
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Creates dummy variables for ALL categories, including rare ones
# Rare categories → many zero columns → near-singular matrix
```

**Solution from cox_survival_v9.py:**
```python
class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups rare categories to 'Other'.

    Frequency threshold: 1% (default)
    Categories with freq < 1% → 'Other'
    """

    def fit(self, X: pd.DataFrame, y=None):
        n = len(X)
        for col in X.columns:
            vc = X[col].value_counts(dropna=False)
            keep = set(vc[vc / n >= self.threshold].index.tolist())
            # Map rare levels to 'Other'
```

**Why this works:**
- Reduces number of dummy variables
- Prevents sparse columns from rare categories
- Improves numerical conditioning

---

## Numerical Stability Analysis

### LAPACK Error Context

The error `LAPACK reported an illegal value in 5-th argument` occurs in:

```
scipy.linalg.lapack → dposv() or dpotrf()
    ↓
Used by CoxPHSurvivalAnalysis for:
    ↓
Solving: H * delta = gradient
where H = Hessian matrix (second derivatives)
```

**When does this fail?**

1. **Singular Hessian (det(H) = 0):**
   - Perfect multicollinearity
   - Zero variance features
   - More features than effective samples

2. **Near-singular Hessian (det(H) ≈ 0):**
   - High multicollinearity
   - Near-zero variance features
   - Ill-conditioned covariance matrix

3. **NaN/Inf values:**
   - Missing values propagated to optimizer
   - Overflow in gradient computation

### Current Framework Vulnerabilities

```python
# Data flow in current framework:
df → split_X_y() → X with potential NaN
    ↓
X → StandardScaler() → X_scaled with NaN if input has NaN
    ↓
X_scaled → OneHotEncoder() → X_transformed with:
    - Many columns from one-hot encoding
    - Rare categories → sparse columns
    - No variance filtering
    ↓
X_transformed → CoxPHSurvivalAnalysis() → Computes Hessian
    ↓
Hessian is singular or near-singular due to:
    - NaN values
    - Zero variance columns
    - Perfect multicollinearity from sparse dummies
    - No regularization to stabilize
    ↓
LAPACK error when attempting to invert/solve
```

---

## Recommended Fixes

### Priority 1: Critical Fixes (Required to Prevent LAPACK Error)

#### Fix 1: Add Missing Value Imputation

**File:** `src/survival_framework/data.py`

```python
from sklearn.impute import SimpleImputer

def make_preprocessor(
    numeric: List[str] = None, categorical: List[str] = None
) -> ColumnTransformer:
    """Create preprocessing pipeline with imputation."""
    numeric = numeric if numeric is not None else NUM_COLS
    categorical = categorical if categorical is not None else CAT_COLS

    # Numeric pipeline with imputation
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline with imputation
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )
    return pre
```

#### Fix 2: Add Variance Threshold Filtering

**File:** `src/survival_framework/data.py`

```python
from sklearn.feature_selection import VarianceThreshold

def make_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Create pipeline with variance filtering."""
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("varth", VarianceThreshold(threshold=1e-12)),  # Remove near-constant features
            ("model", estimator),
        ]
    )
```

#### Fix 3: Add Regularization to CoxPH

**File:** `src/survival_framework/models.py`

```python
@dataclass
class CoxPHWrapper(BaseSurvivalModel):
    """Wrapper for scikit-survival Cox Proportional Hazards model with regularization."""
    name: str = "cox_ph"
    model: CoxPHSurvivalAnalysis = None
    alpha: float = 1.0  # L2 regularization parameter

    def __post_init__(self):
        if self.model is None:
            self.model = CoxPHSurvivalAnalysis(
                alpha=self.alpha,  # Add L2 regularization
                n_iter=200,
                tol=1e-9
            )
```

---

### Priority 2: Recommended Enhancements

#### Enhancement 1: Add Rare Category Grouping

**File:** `src/survival_framework/transformers.py` (NEW)

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Dict, Any, List, Optional

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Groups rare categories to 'Other' for each categorical feature.

    Args:
        threshold: Minimum frequency proportion (0-1). Categories with
                   freq < threshold are grouped to 'Other'.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.level_maps_: Dict[str, Dict[Any, Any]] = {}
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Learn which categories to group."""
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

            self.level_maps_[col] = m
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rare category grouping."""
        X_out = X.copy()
        for col in self.columns_:
            X_out[col] = X_out[col].map(
                lambda x: self.level_maps_[col].get(x, "Other")
            )
        return X_out
```

**Update:** `src/survival_framework/data.py`

```python
from survival_framework.transformers import RareCategoryGrouper

def make_preprocessor(
    numeric: List[str] = None,
    categorical: List[str] = None,
    rare_threshold: float = 0.01
) -> ColumnTransformer:
    """Create preprocessing pipeline with rare category handling."""
    numeric = numeric if numeric is not None else NUM_COLS
    categorical = categorical if categorical is not None else CAT_COLS

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("rare", RareCategoryGrouper(threshold=rare_threshold)),  # NEW
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )
    return pre
```

#### Enhancement 2: Add Feature Selection

**File:** `src/survival_framework/transformers.py`

```python
from lifelines import CoxPHFitter
import numpy as np

class UnivariateCoxSelector(BaseEstimator, TransformerMixin):
    """Univariate feature selection using Cox regression.

    Ranks features by univariate C-index and p-value, selects top k.

    Args:
        k: Number of features to select, or "all" to keep all
        pval_threshold: Maximum p-value for significance
        min_features: Minimum number of features to keep
    """

    def __init__(self, k: Any = "all", pval_threshold: float = 0.05, min_features: int = 5):
        self.k = k
        self.pval_threshold = pval_threshold
        self.min_features = min_features
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Rank features by univariate Cox regression."""
        from sksurv.metrics import concordance_index_censored

        feature_scores = []

        for col in X.columns:
            # Fit univariate Cox model using lifelines
            df_temp = pd.DataFrame({
                'feature': X[col],
                'event': y['event'],
                'time': y['time']
            })

            try:
                cph = CoxPHFitter()
                cph.fit(df_temp, duration_col='time', event_col='event')

                # Get p-value
                pval = cph.summary.loc['feature', 'p']

                # Get C-index
                risk_scores = cph.predict_partial_hazard(df_temp[['feature']])
                c_index = concordance_index_censored(
                    y['event'], y['time'], risk_scores
                )[0]

                feature_scores.append({
                    'feature': col,
                    'c_index': c_index,
                    'pval': pval
                })
            except Exception:
                # Skip features that fail to fit
                continue

        # Filter by p-value
        df_scores = pd.DataFrame(feature_scores)
        df_scores = df_scores[df_scores['pval'] <= self.pval_threshold]

        # Sort by C-index (descending)
        df_scores = df_scores.sort_values('c_index', ascending=False)

        # Select top k
        if self.k == "all":
            k_use = len(df_scores)
        else:
            k_use = max(int(self.k), self.min_features)

        self.selected_features_ = df_scores.head(k_use)['feature'].tolist()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features."""
        return X[self.selected_features_]
```

---

## Testing the Fixes

### Test 1: Verify Imputation Works

```python
# tests/test_data.py

def test_preprocessor_handles_missing_values():
    """Test that preprocessor imputes missing values."""
    df = pd.DataFrame({
        'debit_exp_smooth': [1.0, np.nan, 3.0],
        'typeoftariff_coarse': ['A', np.nan, 'B'],
    })

    preprocessor = make_preprocessor(
        numeric=['debit_exp_smooth'],
        categorical=['typeoftariff_coarse']
    )

    X_transformed = preprocessor.fit_transform(df)

    # Should have no NaN values after imputation
    assert not np.isnan(X_transformed).any()

    # Should have missing indicators
    assert X_transformed.shape[1] > 2  # Original + missing indicators
```

### Test 2: Verify Variance Filtering

```python
def test_pipeline_removes_zero_variance_features():
    """Test that VarianceThreshold removes constant features."""
    df = pd.DataFrame({
        'debit_exp_smooth': [1.0, 2.0, 3.0, 4.0, 5.0],
        'constant_feature': [1.0, 1.0, 1.0, 1.0, 1.0],  # Zero variance
        'typeoftariff_coarse': ['A', 'A', 'B', 'B', 'C'],
    })

    preprocessor = make_preprocessor(
        numeric=['debit_exp_smooth', 'constant_feature'],
        categorical=['typeoftariff_coarse']
    )

    pipeline = make_pipeline(preprocessor, CoxPHWrapper())

    # Should not raise LAPACK error despite zero-variance feature
    # VarianceThreshold should remove it
    y = np.array(
        [(True, 12), (False, 24), (True, 6), (False, 18), (True, 30)],
        dtype=[("event", bool), ("time", float)]
    )

    # This should work without LAPACK error
    pipeline.fit(df, y)
```

### Test 3: Verify Regularization Helps

```python
def test_regularization_prevents_lapack_error(sample_data):
    """Test that regularized CoxPH handles ill-conditioned data."""
    X, y, _ = split_X_y(sample_data)

    # Create ill-conditioned data by duplicating features
    X_duplicated = pd.concat([X, X.add_suffix('_dup')], axis=1)

    preprocessor = make_preprocessor()

    # Without regularization (should potentially fail)
    model_no_reg = CoxPHWrapper(alpha=0)

    # With regularization (should work)
    model_with_reg = CoxPHWrapper(alpha=1.0)

    pipeline_reg = make_pipeline(preprocessor, model_with_reg)

    # This should work even with duplicated (perfectly correlated) features
    pipeline_reg.fit(X_duplicated, y)
```

---

## Implementation Plan

### Step 1: Minimal Fix (PRIORITY)

**Goal:** Stop LAPACK error immediately

**Changes:**
1. Update `data.py::make_preprocessor()` to add `SimpleImputer` with `add_indicator=True`
2. Update `data.py::make_pipeline()` to add `VarianceThreshold`
3. Update `models.py::CoxPHWrapper` to add `alpha=1.0` parameter

**Estimated Time:** 30 minutes
**Testing:** Run main.py on sample data

### Step 2: Enhanced Preprocessing

**Goal:** Match cox_survival_v9.py preprocessing quality

**Changes:**
1. Create `transformers.py` with `RareCategoryGrouper`
2. Update `data.py` to use `RareCategoryGrouper` in categorical pipeline
3. Add `drop="first"` to OneHotEncoder to reduce multicollinearity

**Estimated Time:** 1 hour
**Testing:** Verify reduced feature count, no rare category issues

### Step 3: Feature Selection (Optional)

**Goal:** Add univariate feature screening

**Changes:**
1. Add `UnivariateCoxSelector` to `transformers.py`
2. Create `CoxPHWithSelectionWrapper` model class
3. Add feature selection to pipeline

**Estimated Time:** 2 hours
**Testing:** Compare performance with/without selection

### Step 4: Integration Tests

**Goal:** Ensure fixes work end-to-end

**Changes:**
1. Add `test_lapack_error_fix.py`
2. Test with sample data (2000 rows)
3. Test with missing values
4. Test with rare categories
5. Test with duplicated features

**Estimated Time:** 1 hour

---

## Summary

### Why cox_survival_v9.py Works

✅ **Comprehensive preprocessing:**
- Imputes missing values with indicators
- Groups rare categories
- Filters zero-variance features
- Standardizes numerics

✅ **Feature selection:**
- Univariate Cox screening
- P-value filtering
- Dimensionality reduction

✅ **Regularization:**
- L2 penalty (`alpha=1`)
- Stabilizes ill-conditioned matrices

✅ **Robust pipeline:**
- Multiple safeguards against numerical issues
- Graceful handling of edge cases

### Why Current Framework Fails

❌ **Missing preprocessing:**
- No imputation → NaN values
- No variance filtering → singular matrices
- No rare category handling → sparse columns

❌ **No regularization:**
- `alpha=0` (default) → unstable coefficients
- Sensitive to multicollinearity

❌ **No feature selection:**
- Uses all features → high dimensionality
- No screening for significance

### Critical Path to Fix

**Priority 1 (CRITICAL):**
1. Add `SimpleImputer(add_indicator=True)` to preprocessing
2. Add `VarianceThreshold(threshold=1e-12)` to pipeline
3. Add `alpha=1.0` to `CoxPHSurvivalAnalysis`

**Priority 2 (RECOMMENDED):**
4. Add `RareCategoryGrouper` transformer
5. Add `drop="first"` to OneHotEncoder
6. Add `UnivariateCoxSelector` for feature screening

These changes will align the framework with the working implementation and eliminate the LAPACK error.
