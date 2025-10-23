"""
Integration tests for survival framework end-to-end pipeline.

These tests validate that the complete training workflow executes successfully,
catching issues that unit tests miss (pipeline compatibility, data edge cases, etc.).
"""

import os
import shutil
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="module")
def sample_data_path(project_root):
    """Path to sample data CSV."""
    return project_root / "data" / "sample" / "survival_inputs_sample2000.csv"


@pytest.fixture(scope="module")
def artifacts_dir(project_root):
    """Path to artifacts directory."""
    return project_root / "artifacts"


@pytest.fixture(scope="module")
def models_dir(project_root):
    """Path to models directory."""
    return project_root / "models"


@pytest.fixture(scope="module", autouse=True)
def clean_artifacts(artifacts_dir, models_dir):
    """Clean artifacts before and after module tests."""
    # Clean before
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    if models_dir.exists():
        shutil.rmtree(models_dir)

    yield

    # Keep artifacts after for inspection


class TestEndToEndPipeline:
    """Test complete training pipeline execution."""

    def test_full_pipeline_completes_successfully(self, sample_data_path):
        """Test that full training pipeline completes without errors."""
        from survival_framework.train import train_all_models

        # This should complete without raising exceptions
        train_all_models(str(sample_data_path))

    def test_all_models_train_successfully(self, artifacts_dir):
        """Test that all expected models complete training."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        assert metrics_path.exists(), "model_metrics.csv not created"

        metrics = pd.read_csv(metrics_path)

        # Check all expected models present
        expected_models = {"cox_ph", "coxnet", "weibull_aft", "gbsa", "rsf"}
        actual_models = set(metrics["model"].unique())

        assert expected_models == actual_models, (
            f"Expected models {expected_models}, " f"got {actual_models}"
        )

    def test_all_cv_folds_complete(self, artifacts_dir):
        """Test that all 5 CV folds complete for each model."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)

        # Each model should have 5 folds
        for model in metrics["model"].unique():
            model_folds = metrics[metrics["model"] == model]
            assert len(model_folds) == 5, (
                f"Model {model} should have 5 folds, " f"got {len(model_folds)}"
            )

            # Check fold indices are 0-4
            fold_indices = sorted(model_folds["fold"].unique())
            assert fold_indices == [0, 1, 2, 3, 4], (
                f"Model {model} folds should be [0,1,2,3,4], " f"got {fold_indices}"
            )


class TestArtifactGeneration:
    """Test that all expected artifacts are created."""

    def test_model_metrics_csv_created(self, artifacts_dir):
        """Test that model_metrics.csv is created."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        assert metrics_path.exists(), "model_metrics.csv not created"

        # Check it's valid CSV
        metrics = pd.read_csv(metrics_path)
        assert len(metrics) > 0, "model_metrics.csv is empty"

        # Check required columns
        required_cols = {"model", "fold", "cindex", "ibs", "mean_auc"}
        assert required_cols.issubset(metrics.columns), (
            f"Missing columns: {required_cols - set(metrics.columns)}"
        )

    def test_model_summary_csv_created(self, artifacts_dir):
        """Test that model_summary.csv is created."""
        summary_path = artifacts_dir / "model_summary.csv"
        assert summary_path.exists(), "model_summary.csv not created"

        summary = pd.read_csv(summary_path)
        assert len(summary) > 0, "model_summary.csv is empty"

        # Check aggregated metrics
        required_cols = {"model", "cindex", "ibs", "rank_cindex", "rank_ibs"}
        assert required_cols.issubset(summary.columns), (
            f"Missing columns: {required_cols - set(summary.columns)}"
        )

    def test_ph_flags_csv_created(self, artifacts_dir):
        """Test that PH assumption flags are created."""
        ph_flags_path = artifacts_dir / "ph_flags.csv"
        assert ph_flags_path.exists(), "ph_flags.csv not created"

    def test_model_directories_created(self, artifacts_dir):
        """Test that per-model artifact directories are created."""
        expected_dirs = ["cox_ph", "coxnet", "weibull_aft", "gbsa", "rsf"]

        for model_name in expected_dirs:
            model_dir = artifacts_dir / model_name
            assert model_dir.exists(), f"Directory {model_name} not created"
            assert model_dir.is_dir(), f"{model_name} is not a directory"

    def test_fold_predictions_saved(self, artifacts_dir):
        """Test that per-fold predictions are saved."""
        # Check one model's artifacts
        cox_ph_dir = artifacts_dir / "cox_ph"

        # Should have survival and risk predictions for each fold
        for fold_idx in range(5):
            surv_file = cox_ph_dir / f"cox_ph_fold{fold_idx}_surv.npy"
            risk_file = cox_ph_dir / f"cox_ph_fold{fold_idx}_risk.npy"

            assert surv_file.exists(), f"Survival predictions for fold {fold_idx} not saved"
            assert risk_file.exists(), f"Risk scores for fold {fold_idx} not saved"

            # Check they can be loaded
            surv_pred = np.load(surv_file)
            risk_scores = np.load(risk_file)

            assert surv_pred.shape[0] > 0, f"Empty survival predictions fold {fold_idx}"
            assert risk_scores.shape[0] > 0, f"Empty risk scores fold {fold_idx}"

    def test_trained_models_saved(self, models_dir):
        """Test that trained models are saved."""
        assert models_dir.exists(), "models/ directory not created"

        # Should have one .joblib file per model
        model_files = list(models_dir.glob("*.joblib"))
        assert len(model_files) >= 5, (
            f"Expected 5+ model files, " f"found {len(model_files)}"
        )


class TestMetricsValidation:
    """Test that computed metrics are valid and reasonable."""

    def test_cindex_in_valid_range(self, artifacts_dir):
        """Test that C-index values are in [0.5, 1.0]."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)

        assert (metrics["cindex"] >= 0.5).all(), "C-index below 0.5 (worse than random)"
        assert (metrics["cindex"] <= 1.0).all(), "C-index above 1.0 (invalid)"

    def test_ibs_in_valid_range(self, artifacts_dir):
        """Test that IBS values are in [0.0, 0.25]."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)

        assert (metrics["ibs"] >= 0.0).all(), "IBS below 0.0 (invalid)"
        assert (metrics["ibs"] <= 0.25).all(), "IBS above 0.25 (poor calibration)"

    def test_no_nan_in_metrics(self, artifacts_dir):
        """Test that there are no NaN values in metrics."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)

        assert not metrics["cindex"].isna().any(), "NaN values in C-index"
        assert not metrics["ibs"].isna().any(), "NaN values in IBS"
        assert not metrics["mean_auc"].isna().any(), "NaN values in mean AUC"

    def test_models_perform_better_than_random(self, artifacts_dir):
        """Test that all models achieve C-index > 0.6 (meaningful discrimination)."""
        metrics_path = artifacts_dir / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)

        # Average C-index per model
        avg_cindex = metrics.groupby("model")["cindex"].mean()

        for model, cindex in avg_cindex.items():
            assert cindex > 0.6, (
                f"Model {model} has average C-index {cindex:.3f}, "
                f"expected > 0.6 for meaningful discrimination"
            )


class TestPreprocessingRobustness:
    """Test that preprocessing handles edge cases correctly."""

    def test_preprocessing_handles_missing_values(self):
        """Test that preprocessing pipeline handles missing values."""
        from survival_framework.data import make_preprocessor

        # Create data with missing values
        X = pd.DataFrame(
            {
                "debit_exp_smooth": [1.0, np.nan, 3.0, 4.0, 5.0],
                "typeoftariff_coarse": ["A", np.nan, "B", "A", "B"],
            }
        )

        preprocessor = make_preprocessor(
            numeric=["debit_exp_smooth"], categorical=["typeoftariff_coarse"]
        )

        # Should not raise error
        X_transformed = preprocessor.fit_transform(X)

        # Should have no NaN after transformation
        assert not np.isnan(X_transformed).any(), "NaN values after preprocessing"

    def test_preprocessing_handles_zero_variance(self):
        """Test that variance threshold removes constant features."""
        from survival_framework.data import make_preprocessor, make_pipeline
        from survival_framework.models import CoxPHWrapper

        # Create data with constant feature
        X = pd.DataFrame(
            {
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
                "varying": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": ["A", "B", "A", "B", "A"],
            }
        )

        y = np.array(
            [(True, 12), (False, 24), (True, 6), (False, 18), (True, 30)],
            dtype=[("event", bool), ("time", float)],
        )

        preprocessor = make_preprocessor(numeric=["constant", "varying"], categorical=["cat"])

        pipeline = make_pipeline(preprocessor, CoxPHWrapper())

        # Should not raise error despite constant feature
        pipeline.fit(X, y)

    def test_preprocessing_handles_rare_categories(self):
        """Test that one-hot encoding handles rare categories."""
        from survival_framework.data import make_preprocessor

        # Create data with rare category
        X = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": ["A", "A", "A", "A", "rare"],  # 'rare' appears once
            }
        )

        preprocessor = make_preprocessor(numeric=["numeric"], categorical=["cat"])

        # Should not raise error
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == 5, "Rows lost during preprocessing"


class TestModelCompatibility:
    """Test that all models are compatible with preprocessing pipeline."""

    def test_coxph_compatible_with_pipeline(self):
        """Test that CoxPHWrapper works with preprocessing pipeline."""
        from survival_framework.data import make_preprocessor, make_pipeline
        from survival_framework.models import CoxPHWrapper

        X = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "cat1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        y = np.array(
            [
                (True, 12),
                (False, 24),
                (True, 6),
                (False, 18),
                (True, 30),
                (False, 15),
                (True, 9),
                (False, 21),
                (True, 27),
                (False, 33),
            ],
            dtype=[("event", bool), ("time", float)],
        )

        preprocessor = make_preprocessor(numeric=["num1"], categorical=["cat1"])
        pipeline = make_pipeline(preprocessor, CoxPHWrapper())

        # Should complete without errors
        pipeline.fit(X, y)
        risk_scores = pipeline.predict(X)
        assert len(risk_scores) == len(X)

    def test_coxnet_compatible_with_pipeline(self):
        """Test that CoxnetWrapper works with preprocessing pipeline."""
        from survival_framework.data import make_preprocessor, make_pipeline
        from survival_framework.models import CoxnetWrapper

        X = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "cat1": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            }
        )
        y = np.array(
            [
                (True, 12),
                (False, 24),
                (True, 6),
                (False, 18),
                (True, 30),
                (False, 15),
                (True, 9),
                (False, 21),
                (True, 27),
                (False, 33),
            ],
            dtype=[("event", bool), ("time", float)],
        )

        preprocessor = make_preprocessor(numeric=["num1"], categorical=["cat1"])
        pipeline = make_pipeline(preprocessor, CoxnetWrapper())

        # Should complete without errors
        pipeline.fit(X, y)
        risk_scores = pipeline.predict(X)
        assert len(risk_scores) == len(X)


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Benchmark tests for training performance."""

    def test_training_completes_within_time_limit(self, sample_data_path):
        """Test that training completes within reasonable time (< 5 minutes)."""
        import time
        from survival_framework.train import train_all_models

        start = time.time()
        train_all_models(str(sample_data_path))
        duration = time.time() - start

        # Should complete in under 5 minutes
        assert duration < 300, (
            f"Training took {duration:.1f}s, " f"expected < 300s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
