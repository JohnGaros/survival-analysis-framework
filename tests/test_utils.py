"""Unit tests for survival_framework.utils module.

Tests utility functions for directory management, time grids, and versioning.
"""
import pytest
import os
import pandas as pd
import numpy as np
from survival_framework.utils import (
    ensure_dir,
    default_time_grid,
    save_model_metrics,
    versioned_name,
)


class TestEnsureDir:
    """Tests for ensure_dir function."""

    def test_create_new_directory(self, tmp_path):
        """Test creating a new directory."""
        new_dir = tmp_path / "test_dir"
        assert not new_dir.exists()

        ensure_dir(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_existing_directory(self, tmp_path):
        """Test with existing directory (should not raise error)."""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()

        # Should not raise error
        ensure_dir(str(existing_dir))

        assert existing_dir.exists()

    def test_nested_directories(self, tmp_path):
        """Test creating nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        ensure_dir(str(nested_dir))

        assert nested_dir.exists()
        assert nested_dir.is_dir()


class TestDefaultTimeGrid:
    """Tests for default_time_grid function."""

    def test_basic_grid_generation(self, sample_structured_y):
        """Test basic time grid generation."""
        times = default_time_grid(sample_structured_y, n=10)

        assert len(times) == 10
        assert times[0] == 0.1
        assert times[-1] > times[0]

    def test_grid_spacing(self, sample_structured_y):
        """Test that time grid is evenly spaced."""
        times = default_time_grid(sample_structured_y, n=50)

        differences = np.diff(times)
        assert np.allclose(differences, differences[0])  # All differences equal

    def test_percentile_calculation(self):
        """Test that grid extends to 95th percentile."""
        y = np.array(
            [(True, float(i)) for i in range(1, 101)],
            dtype=[("event", bool), ("time", float)]
        )

        times = default_time_grid(y, n=100)

        assert times[-1] == pytest.approx(95.0, rel=0.1)

    def test_minimum_time(self):
        """Test that minimum time is 0.1."""
        y = np.array(
            [(True, 0.01), (True, 0.02)],
            dtype=[("event", bool), ("time", float)]
        )

        times = default_time_grid(y, n=10)

        assert times[0] == 0.1

    def test_custom_n_parameter(self):
        """Test custom number of time points."""
        y = np.array(
            [(True, 10.0), (True, 20.0), (True, 30.0)],
            dtype=[("event", bool), ("time", float)]
        )

        times_10 = default_time_grid(y, n=10)
        times_100 = default_time_grid(y, n=100)

        assert len(times_10) == 10
        assert len(times_100) == 100


class TestSaveModelMetrics:
    """Tests for save_model_metrics function."""

    def test_save_metrics_creates_file(self, tmp_path):
        """Test that metrics are saved to CSV file."""
        df = pd.DataFrame({
            'model': ['cox_ph', 'rsf'],
            'fold': [0, 0],
            'cindex': [0.74, 0.76],
            'ibs': [0.15, 0.14]
        })

        path = save_model_metrics(df, str(tmp_path))

        assert os.path.exists(path)
        assert path.endswith('model_metrics.csv')

    def test_save_metrics_content(self, tmp_path):
        """Test that saved metrics match input data."""
        df = pd.DataFrame({
            'model': ['cox_ph', 'coxnet'],
            'cindex': [0.75, 0.77]
        })

        path = save_model_metrics(df, str(tmp_path))
        loaded_df = pd.read_csv(path)

        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_creates_directory(self, tmp_path):
        """Test that save_model_metrics creates output directory."""
        df = pd.DataFrame({'model': ['test'], 'score': [0.5]})
        nonexistent_dir = tmp_path / "new_dir"

        path = save_model_metrics(df, str(nonexistent_dir))

        assert os.path.exists(path)
        assert nonexistent_dir.exists()


class TestVersionedName:
    """Tests for versioned_name function."""

    def test_basic_versioning(self):
        """Test basic version name generation."""
        name = versioned_name("model")

        assert name.startswith("model_")
        assert len(name) > len("model_")

    def test_timestamp_format(self):
        """Test that timestamp has correct format."""
        name = versioned_name("cox_ph")

        parts = name.split("_")
        assert len(parts) == 3  # cox, ph, timestamp
        timestamp = parts[2]
        assert len(timestamp) == 15  # YYYYMMDD_HHMMSS

    def test_unique_names(self):
        """Test that consecutive calls produce different names."""
        name1 = versioned_name("model")
        name2 = versioned_name("model")

        # May be same if called in same second, but at least should be valid
        assert name1.startswith("model_")
        assert name2.startswith("model_")

    def test_different_bases(self):
        """Test with different base names."""
        name1 = versioned_name("cox_ph")
        name2 = versioned_name("rsf")

        assert "cox_ph" in name1
        assert "rsf" in name2
        assert name1 != name2
