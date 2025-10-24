from __future__ import annotations
from typing import Tuple, List, Literal
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

# Run type for distinguishing sample vs production runs
RunType = Literal["sample", "production"]

# Expected columns
NUM_COLS = [
    "debit_exp_smooth",
    "credit_exp_smooth",
    "balance_exp_smooth",
    "past_due_balance_exp_smooth",
    "oldest_past_due_exp_smooth",
    "waobd_exp_smooth",
    "kwh_exp_smooth",
    "total_settlements",
    "active_settlements",
    "defaulted_settlements",
]
CAT_COLS = ["typeoftariff_coarse", "risk_level_coarse"]
ID_COL = "account_entities_key"
TIME_COL = "survival_months"
EVENT_COL = "is_terminated"


def load_data(
    file_path: str,
    run_type: RunType = "sample"
) -> pd.DataFrame:
    """Load survival data from CSV or pickle file.

    Automatically detects file format based on extension and loads the data.
    Supports both CSV (.csv) and pickle (.pkl, .pickle) formats.

    Args:
        file_path: Path to input file (CSV or pickle)
        run_type: Type of run - "sample" for development, "production" for full data.
            Used for logging and output organization.

    Returns:
        DataFrame with survival data including features, time, and event columns

    Raises:
        FileNotFoundError: If file_path does not exist
        ValueError: If file format is not supported

    Example:
        >>> # Load CSV
        >>> df = load_data("data/inputs/sample/survival_inputs.csv", run_type="sample")
        >>> print(df.shape)
        (2000, 15)

        >>> # Load pickle
        >>> df = load_data("data/inputs/production/survival_data.pkl", run_type="production")
        >>> print(df.shape)
        (50000, 15)

    Notes:
        - CSV files are loaded with pandas.read_csv()
        - Pickle files are loaded with pandas.read_pickle()
        - run_type does not affect data loading, only metadata logging
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Determine file format from extension
    suffix = file_path.suffix.lower()

    if suffix == '.csv':
        print(f"Loading CSV data from {file_path} (run_type={run_type})")
        df = pd.read_csv(file_path)
    elif suffix in ['.pkl', '.pickle']:
        print(f"Loading pickle data from {file_path} (run_type={run_type})")
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .csv, .pkl, .pickle"
        )

    print(f"Loaded {len(df):,} records with {len(df.columns)} columns")

    # Filter out invalid survival times (must be > 0)
    if TIME_COL in df.columns:
        invalid_count = (df[TIME_COL] <= 0).sum()
        if invalid_count > 0:
            print(f"WARNING: Removing {invalid_count:,} records with survival_months ≤ 0")
            df = df[df[TIME_COL] > 0].copy()
            print(f"Remaining: {len(df):,} records")

    return df


def to_structured_y(df: pd.DataFrame) -> np.ndarray:
    """Create scikit-survival structured array from DataFrame.

    Converts event and time columns to the structured array format required
    by scikit-survival models with named fields ('event', 'time').

    Args:
        df: DataFrame containing EVENT_COL and TIME_COL columns

    Returns:
        Structured numpy array with dtype=[('event', bool), ('time', float)]
        where 'event' indicates whether the event occurred and 'time' is the
        survival time in months

    Example:
        >>> df = pd.DataFrame({'is_terminated': [True, False], 'survival_months': [12.5, 24.0]})
        >>> y = to_structured_y(df)
        >>> y.dtype.names
        ('event', 'time')
    """
    y = np.array(
        list(zip(df[EVENT_COL].astype(bool).values, df[TIME_COL].astype(float).values)),
        dtype=[("event", bool), ("time", float)],
    )
    return y


def make_preprocessor(
    numeric: List[str] = None, categorical: List[str] = None
) -> ColumnTransformer:
    """Create preprocessing pipeline for survival model features.

    Builds a ColumnTransformer that applies imputation, scaling to numeric features
    and imputation, one-hot encoding to categorical features. Missing value indicators
    are added to preserve missingness information.

    Args:
        numeric: List of numeric column names. Defaults to NUM_COLS if None
        categorical: List of categorical column names. Defaults to CAT_COLS if None

    Returns:
        ColumnTransformer configured with:
        - Numeric pipeline: SimpleImputer (median + indicators) -> StandardScaler
        - Categorical pipeline: SimpleImputer (most_frequent + indicators) -> OneHotEncoder

    Example:
        >>> preprocessor = make_preprocessor()
        >>> preprocessor = make_preprocessor(numeric=['feature1', 'feature2'], categorical=['category1'])

    Notes:
        - add_indicator=True creates binary columns for missing values
        - This prevents NaN propagation that can cause LAPACK errors in CoxPH
        - drop='first' in OneHotEncoder reduces multicollinearity
    """
    numeric = numeric if numeric is not None else NUM_COLS
    categorical = categorical if categorical is not None else CAT_COLS

    # Numeric pipeline: impute median + add missing indicators, then scale
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute most frequent + add missing indicators, then one-hot encode
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


def split_X_y(df: pd.DataFrame, dropna: bool = True) -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Extract features, survival labels, and IDs from input DataFrame.

    Splits the input data into feature matrix X, structured survival array y,
    and account IDs. Optionally removes rows with missing values.
    Automatically detects which numeric columns are present in the data.

    Args:
        df: Input DataFrame containing all required columns (NUM_COLS, CAT_COLS,
            ID_COL, TIME_COL, EVENT_COL)
        dropna: If True, remove rows with any missing values. Defaults to True

    Returns:
        Tuple containing:
        - X: DataFrame with shape (n_samples, n_features) containing NUM_COLS + CAT_COLS
        - y: Structured array with shape (n_samples,) and dtype=[('event', bool), ('time', float)]
        - ids: Series with shape (n_samples,) containing account IDs

    Example:
        >>> df = pd.read_csv('data.csv')
        >>> X, y, ids = split_X_y(df)
        >>> X.shape, y.shape, ids.shape
        ((2000, 11), (2000,), (2000,))

    Notes:
        - Only uses numeric columns that are present in the DataFrame
        - kwh_exp_smooth is optional (present in production, absent in sample)
    """
    # Use only numeric columns that are actually present in the data
    num_cols_present = [col for col in NUM_COLS if col in df.columns]

    cols_needed = [ID_COL, TIME_COL, EVENT_COL] + num_cols_present + CAT_COLS
    data = df[cols_needed].copy()
    if dropna:
        data = data.dropna()
    X = data[num_cols_present + CAT_COLS]
    y = to_structured_y(data)
    ids = data[ID_COL]
    return X, y, ids


def make_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """Create sklearn Pipeline combining preprocessing and survival model.

    Constructs a three-step pipeline:
    1. Preprocessing transformation (imputation + scaling/encoding)
    2. Variance threshold filtering (removes near-constant features)
    3. Survival model estimation

    Args:
        preprocessor: ColumnTransformer for feature preprocessing
        estimator: Survival model estimator (e.g., CoxPHWrapper, RSFWrapper)

    Returns:
        Pipeline with steps:
        - 'pre': preprocessing (imputation + scaling/encoding)
        - 'varth': variance threshold filter (removes features with variance < 1e-12)
        - 'model': survival model estimator

    Example:
        >>> preprocessor = make_preprocessor()
        >>> model = CoxPHWrapper()
        >>> pipeline = make_pipeline(preprocessor, model)
        >>> pipeline.fit(X_train, y_train)

    Notes:
        - VarianceThreshold prevents LAPACK errors by removing singular features
        - Threshold of 1e-12 removes effectively constant features
        - This stabilizes matrix operations in CoxPH and other models
    """
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            ("varth", VarianceThreshold(threshold=1e-12)),
            ("model", estimator),
        ]
    )