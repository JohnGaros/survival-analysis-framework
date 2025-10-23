# Survival Analysis Methodology

**Last Updated:** 2025-10-23
**Framework Version:** 1.0

This document describes the theoretical foundations and implementation approach for the survival analysis framework. It is maintained alongside code changes to ensure consistency between theory and implementation.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Survival Analysis Fundamentals](#survival-analysis-fundamentals)
3. [Model Implementations](#model-implementations)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Cross-Validation Strategy](#cross-validation-strategy)
7. [Prediction Methodology](#prediction-methodology)
8. [Implementation Architecture](#implementation-architecture)
9. [References](#references)

---

## Introduction

### Purpose

This framework implements survival analysis for predicting customer churn/termination events. Unlike binary classification (will churn: yes/no) or time-series forecasting, survival analysis jointly models:

1. **Whether** an event will occur (termination)
2. **When** the event will occur (survival time)
3. **Censored observations** where the event has not yet occurred

### Key Advantages

- Handles right-censored data (customers still active at observation end)
- Provides time-dependent risk predictions
- Estimates expected survival time for each customer
- Quantifies uncertainty through survival probabilities

---

## Survival Analysis Fundamentals

### Core Concepts

#### Survival Function S(t)

The survival function represents the probability that an individual survives beyond time t:

```
S(t) = P(T > t)
```

Where:
- T is the random variable representing survival time
- S(t) is monotonically decreasing: S(0) = 1, lim(t→∞) S(t) = 0
- In our context: probability that a customer remains active beyond t months

#### Hazard Function h(t)

The hazard function represents the instantaneous risk of the event occurring at time t, given survival up to t:

```
h(t) = lim(Δt→0) P(t ≤ T < t + Δt | T ≥ t) / Δt
```

Properties:
- h(t) ≥ 0 for all t
- Higher hazard = higher instantaneous risk
- Relationship to survival: S(t) = exp(-∫₀ᵗ h(u)du)

#### Cumulative Hazard Function H(t)

```
H(t) = ∫₀ᵗ h(u)du = -log(S(t))
```

#### Censoring

**Right Censoring** occurs when the event has not been observed by the end of the study period:

- **Event indicator**: δ = 1 if event observed, δ = 0 if censored
- **Observed time**: T_obs = min(T_event, T_censoring)
- Censored observations provide partial information: "survived at least until T_obs"

---

## Model Implementations

### 1. Cox Proportional Hazards (Cox PH)

#### Theory

The Cox PH model assumes the hazard function can be written as:

```
h(t|X) = h₀(t) · exp(β'X)
```

Where:
- h₀(t) is the baseline hazard (unspecified, non-parametric)
- β is the vector of coefficients
- X is the covariate vector
- exp(β'X) is the risk score (hazard ratio)

**Proportional Hazards Assumption**: The ratio of hazards for two individuals is constant over time.

#### Implementation Details

**Module**: `CoxPHWrapper` in `models.py`

- Uses `CoxPHSurvivalAnalysis` from scikit-survival
- **L2 Regularization** (Ridge): α = 1.0 by default to prevent numerical instability
- Optimization: Newton-Raphson with 200 iterations, tolerance 1e-9
- Input validation: checks for finite values, minimum sample size, non-zero variance

**Why L2 Regularization?**
- Prevents LAPACK errors from multicollinearity
- Stabilizes coefficient estimates with high-dimensional data
- Default α=1.0 balances stability and model flexibility

### 2. Elastic Net Cox (Coxnet)

#### Theory

Extends Cox PH with elastic net penalty (L1 + L2 regularization):

```
minimize: -log-likelihood + λ[(1-α)/2 ||β||₂² + α||β||₁]
```

Where:
- λ controls overall regularization strength
- α ∈ [0,1] controls L1/L2 mix
- α=1: Lasso (feature selection)
- α=0: Ridge (shrinkage only)
- 0<α<1: Elastic net (both)

#### Implementation Details

**Module**: `CoxnetWrapper` in `models.py`

- Uses `CoxnetSurvivalAnalysis` from scikit-survival
- Default: l1_ratio=0.5 (balanced L1/L2)
- Automatic regularization path selection via cross-validation
- Feature selection through L1 penalty

### 3. Weibull Accelerated Failure Time (AFT)

#### Theory

AFT models assume covariates act multiplicatively on survival time:

```
log(T) = β'X + σε
```

Where:
- ε follows a specified distribution (Weibull in our case)
- σ is a scale parameter
- exp(β'X) is the acceleration factor

Weibull distribution:
```
S(t|X) = exp(-(λt)ᵖ) where λ = exp(-β'X/σ)
```

- p > 1: increasing hazard (aging)
- p < 1: decreasing hazard
- p = 1: constant hazard (exponential)

#### Implementation Details

**Module**: `WeibullAFTWrapper` in `models.py`

- Uses `WeibullAFTFitter` from lifelines
- Parametric model with interpretable acceleration factors
- Requires DataFrame input with 'time' and 'event' columns
- Directly models survival time distribution

### 4. Gradient Boosting Survival Analysis (GBSA)

#### Theory

Ensemble method that builds an additive model:

```
h(t|X) = h₀(t) · exp(∑ᵢ fᵢ(X))
```

Where each fᵢ is a regression tree fitted to negative gradients of the partial likelihood loss.

**Advantages**:
- Captures non-linear relationships
- Handles interactions automatically
- Robust to irrelevant features
- No proportional hazards assumption

#### Implementation Details

**Module**: `GBSAWrapper` in `models.py`

- Uses `GradientBoostingSurvivalAnalysis` from scikit-survival
- Tree-based ensemble with boosting
- Flexible non-parametric approach
- Often achieves best predictive performance

### 5. Random Survival Forest (RSF)

#### Theory

Extension of random forests to survival data:

1. Bootstrap sample from training data
2. At each node, randomly select subset of features
3. Split node to maximize survival difference (log-rank test)
4. Aggregate predictions across trees

**Cumulative Hazard Estimate**:
```
Ĥ(t|X) = average over trees of cumulative hazard in terminal node
```

#### Implementation Details

**Module**: `RSFWrapper` in `models.py`

- Uses `RandomSurvivalForest` from scikit-survival
- Ensemble of survival trees
- Handles non-linear patterns and interactions
- Provides feature importance measures

---

## Data Preprocessing Pipeline

### Pipeline Architecture

The preprocessing pipeline (`make_pipeline()` in `data.py`) consists of three stages:

```
Input Data → Preprocessing → Variance Filtering → Survival Model
```

### Stage 1: Column Transformation

**Numeric Features**:
```python
Pipeline([
    SimpleImputer(strategy='median', add_indicator=True),
    StandardScaler()
])
```

- **Imputation**: Missing values replaced with median
- **Missing indicators**: Binary flags for missingness (preserves information)
- **Scaling**: Zero mean, unit variance (required for Cox models)

**Categorical Features**:
```python
Pipeline([
    SimpleImputer(strategy='most_frequent', add_indicator=True),
    OneHotEncoder(drop='first', handle_unknown='ignore')
])
```

- **Imputation**: Missing values replaced with mode
- **Missing indicators**: Binary flags for missingness
- **One-hot encoding**: drop='first' to avoid multicollinearity
- **Unknown handling**: Assigns all-zero vector for new categories

### Stage 2: Variance Threshold

```python
VarianceThreshold(threshold=1e-12)
```

**Purpose**: Remove effectively constant features that cause:
- Singular covariance matrices
- LAPACK numerical errors
- Unstable coefficient estimates

**Rationale**: Features with variance < 1e-12 provide no discriminative information and destabilize matrix operations in Cox models.

### Stage 3: Survival Model

The fitted model from Stage 1-2 is passed to the survival estimator.

### Why This Design?

1. **Missing value handling**: Prevents NaN propagation while preserving missingness as a signal
2. **Numerical stability**: Scaling + variance filtering prevent matrix singularity
3. **Regularization compatibility**: Scaled features enable fair L1/L2 penalties
4. **Censoring-aware**: All stages preserve sample indices for proper y alignment

---

## Evaluation Metrics

### 1. Concordance Index (C-index)

#### Definition

Measures the proportion of all comparable pairs where predictions and outcomes are concordant:

```
C-index = P(risk_i > risk_j | time_i < time_j, event_i = 1)
```

**Interpretation**:
- 0.5: Random predictions
- 1.0: Perfect discrimination
- >0.7: Generally considered good
- Analogous to AUC-ROC for survival data

#### Implementation

- Uses Harrell's C-index with IPCW (inverse probability of censoring weighting)
- Handles censored observations appropriately
- Primary metric for model comparison

### 2. Integrated Brier Score (IBS)

#### Definition

Measures prediction accuracy across time by averaging Brier scores:

```
BS(t) = (1/n) ∑ᵢ [S(t|Xᵢ) - I(Tᵢ > t)]² · W(Tᵢ, δᵢ, t)
IBS = ∫ BS(t) dt / (t_max - t_min)
```

Where W is the IPCW weight accounting for censoring.

**Interpretation**:
- 0: Perfect calibration
- Lower is better
- Measures both discrimination and calibration
- Time-dependent metric

#### Implementation

- Evaluated at time horizons: 3, 6, 12, 18, 24 months
- Uses IPCW to handle censoring
- Secondary metric for model selection

### 3. Time-Dependent AUC

#### Definition

Extension of ROC AUC for time-varying predictions:

```
AUC(t) = P(risk_i > risk_j | Tᵢ = t, Tⱼ > t)
```

**Interpretation**:
- Discrimination at specific time point t
- 0.5: Random classifier
- 1.0: Perfect discrimination
- Evaluated at multiple time horizons

#### Implementation

- Uses cumulative/dynamic AUC definition
- Evaluated at CV time horizons
- Provides time-specific performance insights

---

## Cross-Validation Strategy

### Event-Balanced Stratified K-Fold

#### Motivation

Standard K-fold can create folds with:
- Imbalanced event rates (e.g., 5% vs 15%)
- Insufficient events for model fitting
- Biased performance estimates

#### Implementation

**Function**: `event_balanced_splitter()` in `validation.py`

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**Stratification variable**: Event indicator (δ)

**Ensures**:
- Each fold has approximately equal proportion of events
- Both censored and event samples in train/test
- Representative folds for unbiased evaluation

### Fold-Aware Time Grid Constraints

**Problem**: Survival metrics require time points within the test set's observed range.

**Solution**: `default_time_grid()` with test set constraints:

```python
t_min = max(train_5th_percentile, test_min + 0.1)
t_max = min(train_95th_percentile, test_max - 0.1)
```

**Benefits**:
- Prevents extrapolation errors
- Avoids "all times must be within follow-up" errors
- Ensures valid metric computation

### Cross-Validation Workflow

For each model:
1. Create 5 event-balanced folds
2. For each fold:
   - Fit preprocessing + model on training data
   - Generate fold-aware time grid
   - Predict on test fold
   - Compute C-index, IBS, time-dependent AUC
   - Save predictions to `artifacts/<model>/<model>_fold{i}_*.npy`
3. Aggregate metrics across folds
4. Rank models by mean C-index (primary), IBS (secondary)

---

## Prediction Methodology

### Model Selection

**Criterion**: Highest mean C-index from cross-validation

**Process**:
1. Read `artifacts/model_summary.csv`
2. Select top-ranked model (sorted by rank_cindex, rank_ibs)
3. Load most recent saved model from `models/`

### Survival Probability Predictions

**Point Estimates at Fixed Horizons**:

Given feature vector X and time point t:
```
Ŝ(t|X) = model.predict_survival_function(X, times=[t])
```

**Current horizons**: 3, 6, 12, 18, 24, 36 months

**Interpretation**:
- P(customer remains active beyond t months | features X)
- Higher probability = lower churn risk
- Used for threshold-based interventions (e.g., contact if Ŝ(12) < 0.7)

### Expected Survival Time

**Restricted Mean Survival Time (RMST)**:

```
E[T|X] = ∫₀ᵗᵐᵃˣ Ŝ(u|X) du
```

**Implementation**:

```python
def calculate_expected_survival_time(pipeline, X):
    times = np.linspace(0, t_max, 200)
    surv_probs = pipeline.predict_survival_function(X, times)
    expected_time = trapezoid(surv_probs, times)  # area under curve
```

**Where**:
- t_max = maximum observed time in training data (auto-detected)
- 200 points for numerical integration (trapezoidal rule)

**Properties**:
- **Intuitive interpretation**: Average months until termination
- **Single summary metric**: Easier for business stakeholders than probability curves
- **Restricted**: Lower bound if survival curve doesn't reach 0
- **Higher value = lower risk**: Directly ranks customers by expected retention

**Use Cases**:
- Customer lifetime value estimation
- Prioritizing retention interventions
- Segmentation by risk level

### Output Format

Each prediction CSV contains per-account:

| Column | Description | Range |
|--------|-------------|-------|
| `account_entities_key` | Account identifier | - |
| `model_name` | Best model used | - |
| `expected_survival_months` | RMST | [0, t_max] |
| `survival_prob_3m` | P(survive > 3 months) | [0, 1] |
| `survival_prob_6m` | P(survive > 6 months) | [0, 1] |
| `survival_prob_12m` | P(survive > 12 months) | [0, 1] |
| `survival_prob_18m` | P(survive > 18 months) | [0, 1] |
| `survival_prob_24m` | P(survive > 24 months) | [0, 1] |
| `survival_prob_36m` | P(survive > 36 months) | [0, 1] |

---

## Implementation Architecture

### Module Structure

```
survival_framework/
├── data.py           # Data loading, preprocessing pipelines
├── models.py         # Survival model wrappers
├── validation.py     # Cross-validation, evaluation
├── metrics.py        # Survival-specific metrics
├── predict.py        # Prediction generation
├── train.py          # Training orchestration
├── tracking.py       # MLflow experiment tracking
└── utils.py          # Helper utilities
```

### Design Principles

1. **Unified Interface**: All models inherit from `BaseSurvivalModel` with standard `fit()`, `predict_survival_function()`, `score()` methods

2. **Pipeline Composition**: sklearn Pipeline pattern for reproducible preprocessing + modeling

3. **Separation of Concerns**:
   - `data.py`: Feature engineering
   - `models.py`: Algorithmic logic
   - `validation.py`: Evaluation protocol
   - `train.py`: Workflow orchestration

4. **Type Safety**: Type hints on all function signatures

5. **Comprehensive Docstrings**: Google-style docstrings with Args, Returns, Examples

6. **Experiment Tracking**: MLflow integration for reproducibility

### Key Abstractions

#### Structured Array for Survival Data

```python
y = np.array(
    [(event_1, time_1), (event_2, time_2), ...],
    dtype=[('event', bool), ('time', float)]
)
```

**Rationale**:
- scikit-survival standard format
- Type-safe event/time pairing
- Prevents index misalignment

#### Model Wrapper Pattern

```python
class BaseSurvivalModel:
    def fit(self, X, y) -> self
    def predict_survival_function(self, X, times) -> np.ndarray
    def score(self, X, y) -> float
```

**Benefits**:
- Consistent API across libraries (scikit-survival, lifelines)
- Easy to add new models
- Enables pipeline composition

---

## References

### Theoretical Foundations

1. **Cox, D. R. (1972)**. "Regression Models and Life-Tables." *Journal of the Royal Statistical Society*, Series B, 34(2), 187-220.

2. **Klein, J. P., & Moeschberger, M. L. (2003)**. *Survival Analysis: Techniques for Censored and Truncated Data* (2nd ed.). Springer.

3. **Ishwaran, H., et al. (2008)**. "Random Survival Forests." *The Annals of Applied Statistics*, 2(3), 841-860.

4. **Royston, P., & Altman, D. G. (2013)**. "External validation of a Cox prognostic model: principles and methods." *BMC Medical Research Methodology*, 13, 33.

### Implementation References

5. **Pölsterl, S. (2020)**. "scikit-survival: A Library for Time-to-Event Analysis Built on Top of scikit-learn." *Journal of Machine Learning Research*, 21(212), 1-6.

6. **Davidson-Pilon, C. (2019)**. *lifelines: survival analysis in Python*. Journal of Open Source Software, 4(40), 1317.

7. **Harrell, F. E., et al. (1996)**. "Multivariable prognostic models: issues in developing models, evaluating assumptions and adequacy, and measuring and reducing errors." *Statistics in Medicine*, 15(4), 361-387.

---

## Changelog

### 2025-10-23 - Initial Version
- Documented survival analysis theory and fundamentals
- Described all five model implementations (Cox PH, Coxnet, Weibull AFT, GBSA, RSF)
- Detailed preprocessing pipeline with L2 regularization and variance filtering
- Explained evaluation metrics (C-index, IBS, time-dependent AUC)
- Documented event-balanced cross-validation strategy
- Added prediction methodology including RMST calculation
- Outlined implementation architecture and design principles

---

**Maintenance Note**: This document should be updated whenever:
- New models are added to the framework
- Preprocessing steps are modified
- Evaluation metrics are changed
- Prediction outputs are altered
- Theoretical assumptions are revised
