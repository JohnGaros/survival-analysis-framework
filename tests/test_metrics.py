"""Unit tests for survival_framework.metrics module.

Tests survival-specific metrics including C-index, IBS, and time-dependent AUC.
"""
import pytest
import numpy as np
from survival_framework.metrics import (
    compute_cindex,
    compute_ibs,
    compute_time_dependent_auc,
)


@pytest.fixture
def simple_survival_data():
    """Create simple survival data for testing metrics.

    Returns:
        Tuple of (y_train, y_test, risk_scores, surv_pred, times)
    """
    y_train = np.array(
        [(True, 10.0), (False, 20.0), (True, 5.0), (False, 25.0), (True, 15.0)] * 10,
        dtype=[("event", bool), ("time", float)]
    )

    y_test = np.array(
        [(True, 12.0), (False, 22.0), (True, 8.0), (False, 18.0), (True, 14.0)],
        dtype=[("event", bool), ("time", float)]
    )

    # Higher risk scores for shorter survival times
    risk_scores = np.array([0.8, 0.2, 0.9, 0.3, 0.7])

    # Survival probabilities at different time points (5 samples, 4 time points)
    times = np.array([5.0, 10.0, 15.0, 20.0])
    surv_pred = np.array([
        [0.9, 0.7, 0.5, 0.3],  # Sample 1
        [0.95, 0.9, 0.85, 0.8],  # Sample 2
        [0.8, 0.6, 0.4, 0.2],  # Sample 3
        [0.96, 0.92, 0.88, 0.84],  # Sample 4
        [0.85, 0.7, 0.55, 0.4],  # Sample 5
    ])

    return y_train, y_test, risk_scores, surv_pred, times


class TestComputeCindex:
    """Tests for compute_cindex function."""

    def test_cindex_range(self, simple_survival_data):
        """Test that C-index is in valid range [0.5, 1.0]."""
        y_train, y_test, risk_scores, _, _ = simple_survival_data

        cindex = compute_cindex(y_train, y_test, risk_scores)

        assert 0.0 <= cindex <= 1.0
        assert isinstance(cindex, float)

    def test_perfect_discrimination(self):
        """Test C-index with perfect risk ordering."""
        y_train = np.array(
            [(True, 10.0), (True, 20.0), (True, 30.0)] * 5,
            dtype=[("event", bool), ("time", float)]
        )
        y_test = np.array(
            [(True, 5.0), (True, 15.0), (True, 25.0)],
            dtype=[("event", bool), ("time", float)]
        )
        # Perfect ordering: highest risk for shortest time
        risk_scores = np.array([1.0, 0.5, 0.1])

        cindex = compute_cindex(y_train, y_test, risk_scores)

        assert cindex > 0.8  # Should be high with perfect ordering

    def test_cindex_with_censoring(self):
        """Test C-index computation with censored observations."""
        y_train = np.array(
            [(True, 10.0), (False, 20.0), (True, 15.0)] * 5,
            dtype=[("event", bool), ("time", float)]
        )
        y_test = np.array(
            [(True, 12.0), (False, 25.0), (True, 8.0)],
            dtype=[("event", bool), ("time", float)]
        )
        risk_scores = np.array([0.6, 0.2, 0.9])

        cindex = compute_cindex(y_train, y_test, risk_scores)

        assert isinstance(cindex, float)
        assert not np.isnan(cindex)


class TestComputeIBS:
    """Tests for compute_ibs function."""

    def test_ibs_range(self, simple_survival_data):
        """Test that IBS is non-negative."""
        y_train, y_test, _, surv_pred, times = simple_survival_data

        ibs = compute_ibs(times, y_train, y_test, surv_pred)

        assert ibs >= 0.0
        assert isinstance(ibs, float)

    def test_ibs_shape_validation(self, simple_survival_data):
        """Test IBS with correct prediction shape."""
        y_train, y_test, _, surv_pred, times = simple_survival_data

        assert surv_pred.shape == (len(y_test), len(times))

        ibs = compute_ibs(times, y_train, y_test, surv_pred)

        assert not np.isnan(ibs)

    def test_ibs_perfect_predictions(self):
        """Test IBS with near-perfect survival predictions."""
        y_train = np.array(
            [(True, 10.0), (False, 20.0)] * 10,
            dtype=[("event", bool), ("time", float)]
        )
        y_test = np.array(
            [(True, 8.0), (False, 25.0)],
            dtype=[("event", bool), ("time", float)]
        )
        times = np.array([5.0, 10.0, 15.0, 20.0])

        # Near-perfect predictions
        surv_pred = np.array([
            [0.99, 0.5, 0.01, 0.001],  # Dies at 8 months
            [0.99, 0.98, 0.97, 0.96],  # Survives beyond 25 months
        ])

        ibs = compute_ibs(times, y_train, y_test, surv_pred)

        assert ibs < 0.5  # Should be relatively low


class TestComputeTimeDependentAUC:
    """Tests for compute_time_dependent_auc function."""

    def test_auc_output_shape(self, simple_survival_data):
        """Test that AUC returns correct output shapes."""
        y_train, y_test, risk_scores, _, times = simple_survival_data

        aucs, mean_auc = compute_time_dependent_auc(y_train, y_test, times, risk_scores)

        assert len(aucs) == len(times)
        assert isinstance(mean_auc, (float, np.floating))

    def test_auc_range(self, simple_survival_data):
        """Test that AUC values are in valid range."""
        y_train, y_test, risk_scores, _, times = simple_survival_data

        aucs, mean_auc = compute_time_dependent_auc(y_train, y_test, times, risk_scores)

        assert np.all((aucs >= 0.0) & (aucs <= 1.0))
        assert 0.0 <= mean_auc <= 1.0

    def test_auc_with_good_discrimination(self):
        """Test AUC with well-discriminating risk scores."""
        y_train = np.array(
            [(True, 10.0), (False, 20.0), (True, 15.0)] * 10,
            dtype=[("event", bool), ("time", float)]
        )
        y_test = np.array(
            [(True, 5.0), (True, 12.0), (False, 25.0), (True, 18.0)],
            dtype=[("event", bool), ("time", float)]
        )
        # Risk scores inversely related to survival time
        risk_scores = np.array([1.0, 0.7, 0.1, 0.5])
        times = np.array([6.0, 12.0, 18.0])

        aucs, mean_auc = compute_time_dependent_auc(y_train, y_test, times, risk_scores)

        assert mean_auc > 0.5  # Should be better than random
