"""Unit tests for survival_framework.data module.

Tests data loading, preprocessing, and pipeline construction functionality.
"""
import pytest
import numpy as np
import pandas as pd
from survival_framework.data import (
    to_structured_y,
    make_preprocessor,
    split_X_y,
    make_pipeline,
    NUM_COLS,
    CAT_COLS,
)
from survival_framework.models import CoxPHWrapper


class TestToStructuredY:
    """Tests for to_structured_y function."""

    def test_basic_conversion(self):
        """Test basic conversion of DataFrame to structured array."""
        df = pd.DataFrame({
            'is_terminated': [True, False, True],
            'survival_months': [12.5, 24.0, 6.0]
        })
        y = to_structured_y(df)

        assert y.dtype.names == ('event', 'time')
        assert len(y) == 3
        assert y['event'][0] == True
        assert y['time'][1] == 24.0

    def test_boolean_event_conversion(self):
        """Test that events are properly converted to boolean."""
        df = pd.DataFrame({
            'is_terminated': [1, 0, 1],
            'survival_months': [10.0, 20.0, 15.0]
        })
        y = to_structured_y(df)

        assert y['event'].dtype == bool
        assert y['event'][0] == True
        assert y['event'][1] == False

    def test_time_float_conversion(self):
        """Test that times are properly converted to float."""
        df = pd.DataFrame({
            'is_terminated': [True, False],
            'survival_months': [12, 24]
        })
        y = to_structured_y(df)

        assert y['time'].dtype == np.float64
        assert y['time'][0] == 12.0


class TestMakePreprocessor:
    """Tests for make_preprocessor function."""

    def test_default_columns(self):
        """Test preprocessor with default column names."""
        preprocessor = make_preprocessor()

        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2
        assert preprocessor.transformers[0][0] == 'num'
        assert preprocessor.transformers[1][0] == 'cat'

    def test_custom_columns(self):
        """Test preprocessor with custom column names."""
        custom_num = ['feature1', 'feature2']
        custom_cat = ['category1']

        preprocessor = make_preprocessor(numeric=custom_num, categorical=custom_cat)

        assert preprocessor.transformers[0][2] == custom_num
        assert preprocessor.transformers[1][2] == custom_cat

    def test_fit_transform(self, small_sample_data):
        """Test that preprocessor can fit and transform data."""
        preprocessor = make_preprocessor()
        X = small_sample_data[NUM_COLS + CAT_COLS]

        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] > len(NUM_COLS)  # One-hot encoded cats


class TestSplitXY:
    """Tests for split_X_y function."""

    def test_basic_split(self, sample_data):
        """Test basic splitting of data."""
        X, y, ids = split_X_y(sample_data)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        assert isinstance(ids, pd.Series)
        assert len(X) == len(y) == len(ids)

    def test_feature_columns(self, sample_data):
        """Test that correct feature columns are extracted."""
        X, y, ids = split_X_y(sample_data)

        expected_cols = NUM_COLS + CAT_COLS
        assert list(X.columns) == expected_cols

    def test_structured_array_format(self, sample_data):
        """Test that y has correct structured array format."""
        X, y, ids = split_X_y(sample_data)

        assert y.dtype.names == ('event', 'time')
        assert y['event'].dtype == bool
        assert y['time'].dtype == np.float64

    def test_dropna_true(self, sample_data):
        """Test that dropna=True removes missing values."""
        # Add some missing values
        df = sample_data.copy()
        df.loc[0, 'debit_exp_smooth'] = np.nan

        X, y, ids = split_X_y(df, dropna=True)

        assert len(X) < len(sample_data)
        assert not X.isnull().any().any()

    def test_dropna_false(self, sample_data):
        """Test that dropna=False preserves rows with NaN."""
        df = sample_data.copy()
        original_len = len(df)
        df.loc[0, 'debit_exp_smooth'] = np.nan

        X, y, ids = split_X_y(df, dropna=False)

        assert len(X) == original_len


class TestMakePipeline:
    """Tests for make_pipeline function."""

    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        preprocessor = make_preprocessor()
        model = CoxPHWrapper()

        pipeline = make_pipeline(preprocessor, model)

        assert 'pre' in pipeline.named_steps
        assert 'model' in pipeline.named_steps

    def test_pipeline_fit(self, small_sample_data):
        """Test that pipeline can fit data."""
        X, y, ids = split_X_y(small_sample_data)
        preprocessor = make_preprocessor()
        model = CoxPHWrapper()
        pipeline = make_pipeline(preprocessor, model)

        pipeline.fit(X, y)

        assert hasattr(pipeline.named_steps['pre'], 'transform')
        assert hasattr(pipeline.named_steps['model'], 'predict_survival_function')
