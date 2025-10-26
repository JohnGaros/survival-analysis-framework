"""Tests for stratum-level aggregate statistics module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from survival_framework.strata_analysis import (
    create_stratum_identifier,
    compute_strata_summary,
    compute_strata_predictions,
    save_strata_artifacts,
)
from survival_framework.data import to_structured_y


@pytest.fixture
def sample_strata_data():
    """Create sample data with categorical strata columns."""
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'typeoftariff_coarse': np.random.choice(['tariff_A', 'tariff_B'], n),
        'risk_level_coarse': np.random.choice(['low', 'high'], n),
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'survival_months': np.random.uniform(1, 36, n),
        'is_terminated': np.random.choice([True, False], n, p=[0.3, 0.7])
    })

    X = df[['typeoftariff_coarse', 'risk_level_coarse', 'feature1', 'feature2']]
    y = to_structured_y(df)

    return X, y, df


def test_create_stratum_identifier(sample_strata_data):
    """Test stratum identifier creation."""
    X, y, df = sample_strata_data

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    strata = create_stratum_identifier(X, strata_cols)

    # Check return type
    assert isinstance(strata, pd.Series)
    assert strata.name == 'stratum'

    # Check format (should be "value1|value2")
    assert all('|' in s for s in strata)

    # Check unique values
    unique_strata = strata.unique()
    assert len(unique_strata) <= 4  # Max 2x2 combinations

    # Verify specific examples
    mask = (X['typeoftariff_coarse'] == 'tariff_A') & (X['risk_level_coarse'] == 'low')
    if mask.any():
        assert all(strata[mask] == 'tariff_A|low')


def test_compute_strata_summary(sample_strata_data):
    """Test stratum summary statistics computation."""
    X, y, df = sample_strata_data

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    summary = compute_strata_summary(X, y, strata_cols)

    # Check return type and structure
    assert isinstance(summary, pd.DataFrame)
    assert 'stratum' in summary.columns

    # Check required columns
    required_cols = [
        'stratum', 'typeoftariff_coarse', 'risk_level_coarse',
        'n_samples', 'n_events', 'event_rate',
        'mean_survival_time', 'median_survival_time',
        'min_survival_time', 'max_survival_time'
    ]
    for col in required_cols:
        assert col in summary.columns, f"Missing column: {col}"

    # Check data validity
    assert all(summary['n_samples'] > 0)
    assert all(summary['n_events'] >= 0)
    assert all((summary['event_rate'] >= 0) & (summary['event_rate'] <= 1))
    assert all(summary['mean_survival_time'] > 0)

    # Check totals
    assert summary['n_samples'].sum() == len(X)
    assert summary['n_events'].sum() == y['event'].sum()


def test_compute_strata_predictions(sample_strata_data):
    """Test stratum-level prediction aggregation."""
    X, y, df = sample_strata_data

    # Simulate test fold data
    n_test = 30
    test_indices = np.arange(n_test)
    risk_scores = np.random.randn(n_test)
    times = np.array([6, 12, 24])
    surv_probs = np.random.uniform(0.5, 1.0, size=(n_test, len(times)))

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    strata_preds = compute_strata_predictions(
        X=X,
        y_struct=y,
        strata_cols=strata_cols,
        test_indices=test_indices,
        risk_scores=risk_scores,
        surv_probs=surv_probs,
        times=times,
        model_name='test_model',
        fold_idx=0
    )

    # Check return type and structure
    assert isinstance(strata_preds, pd.DataFrame)
    assert 'model' in strata_preds.columns
    assert 'fold' in strata_preds.columns
    assert 'stratum' in strata_preds.columns

    # Check model and fold identifiers
    assert all(strata_preds['model'] == 'test_model')
    assert all(strata_preds['fold'] == 0)

    # Check risk statistics columns
    assert 'n_samples' in strata_preds.columns
    assert 'risk_mean' in strata_preds.columns
    assert 'risk_std' in strata_preds.columns

    # Check survival probability columns
    for t in times:
        assert f'surv_{int(t)}mo_mean' in strata_preds.columns
        assert f'surv_{int(t)}mo_std' in strata_preds.columns

    # Check totals
    assert strata_preds['n_samples'].sum() == n_test


def test_compute_strata_predictions_without_surv_probs(sample_strata_data):
    """Test stratum predictions when survival probabilities are not available."""
    X, y, df = sample_strata_data

    n_test = 30
    test_indices = np.arange(n_test)
    risk_scores = np.random.randn(n_test)
    times = np.array([6, 12, 24])

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    strata_preds = compute_strata_predictions(
        X=X,
        y_struct=y,
        strata_cols=strata_cols,
        test_indices=test_indices,
        risk_scores=risk_scores,
        surv_probs=None,  # No survival probabilities
        times=times,
        model_name='test_model',
        fold_idx=0
    )

    # Should still work with just risk scores
    assert isinstance(strata_preds, pd.DataFrame)
    assert 'risk_mean' in strata_preds.columns

    # Should not have survival probability columns
    for t in times:
        assert f'surv_{int(t)}mo_mean' not in strata_preds.columns


def test_save_strata_artifacts(sample_strata_data, tmp_path):
    """Test saving stratum artifacts to disk."""
    X, y, df = sample_strata_data

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    artifacts_path = str(tmp_path)

    paths = save_strata_artifacts(X, y, strata_cols, artifacts_path)

    # Check return value
    assert isinstance(paths, dict)
    assert 'strata_summary' in paths

    # Check file was created
    summary_path = Path(paths['strata_summary'])
    assert summary_path.exists()

    # Check file contents
    summary = pd.read_csv(summary_path)
    assert len(summary) > 0
    assert 'stratum' in summary.columns
    assert 'n_samples' in summary.columns


def test_strata_identifier_single_column(sample_strata_data):
    """Test stratum identifier with single categorical column."""
    X, y, df = sample_strata_data

    strata_cols = ['typeoftariff_coarse']
    strata = create_stratum_identifier(X, strata_cols)

    # Should work with single column (no pipe separator needed)
    assert isinstance(strata, pd.Series)
    unique_strata = strata.unique()
    assert len(unique_strata) <= 2  # Max 2 tariff types


def test_strata_summary_edge_cases(sample_strata_data):
    """Test stratum summary with edge cases."""
    X, y, df = sample_strata_data

    # Test with stratum that has no events
    X_modified = X.copy()
    y_modified = y.copy()

    # Force first 10 samples to be tariff_A|low with no events
    X_modified.iloc[:10, X_modified.columns.get_loc('typeoftariff_coarse')] = 'tariff_A'
    X_modified.iloc[:10, X_modified.columns.get_loc('risk_level_coarse')] = 'low'
    y_modified['event'][:10] = False

    strata_cols = ['typeoftariff_coarse', 'risk_level_coarse']
    summary = compute_strata_summary(X_modified, y_modified, strata_cols)

    # Should handle stratum with 0 events
    zero_event_strata = summary[summary['n_events'] == 0]
    if len(zero_event_strata) > 0:
        assert all(zero_event_strata['event_rate'] == 0)
