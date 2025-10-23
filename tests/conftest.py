"""Pytest configuration and shared fixtures for survival framework tests.

This module provides fixtures for loading test data, creating temporary
directories, and setting up test environments.
"""
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory.

    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_data_path(project_root):
    """Return path to sample CSV data.

    Args:
        project_root: Project root directory fixture

    Returns:
        Path: Path to sample CSV file
    """
    return project_root / "data" / "sample" / "survival_inputs_sample2000.csv"


@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """Return path to test output directory.

    Args:
        project_root: Project root directory fixture

    Returns:
        Path: Path to test output directory
    """
    output_dir = project_root / "data" / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_data(sample_data_path):
    """Load sample survival data.

    Args:
        sample_data_path: Path to sample CSV fixture

    Returns:
        pd.DataFrame: Loaded sample data
    """
    return pd.read_csv(sample_data_path)


@pytest.fixture
def small_sample_data(sample_data):
    """Return small subset of sample data for faster tests.

    Args:
        sample_data: Full sample data fixture

    Returns:
        pd.DataFrame: First 100 rows of sample data
    """
    return sample_data.head(100).copy()


@pytest.fixture
def sample_structured_y():
    """Create small structured survival array for testing.

    Returns:
        np.ndarray: Structured array with dtype=[('event', bool), ('time', float)]
    """
    return np.array(
        [(True, 12.5), (False, 24.0), (True, 6.0), (False, 18.0), (True, 30.0)],
        dtype=[("event", bool), ("time", float)]
    )


@pytest.fixture
def temp_artifacts_dir(tmp_path):
    """Create temporary directory for test artifacts.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path: Temporary artifacts directory
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return artifacts_dir


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create temporary directory for test models.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path: Temporary models directory
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


@pytest.fixture(autouse=True)
def cleanup_mlflow_runs():
    """Clean up MLflow tracking URIs after each test.

    Ensures tests don't interfere with each other's MLflow tracking.
    """
    import mlflow
    yield
    # Reset tracking URI after each test
    mlflow.set_tracking_uri(None)
