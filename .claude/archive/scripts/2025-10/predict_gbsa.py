"""Generate production predictions using the GBSA model.

This script directly loads the GBSA model (best model from sample run)
and generates survival probability predictions for production data.

Usage:
    python scripts/predict_gbsa.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from survival_framework.data import load_data, split_X_y, make_preprocessor
from survival_framework.predict import calculate_expected_survival_time
from survival_framework.utils import get_output_paths
from sklearn.pipeline import Pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Generate production predictions using GBSA model."""
    logger.info("="*70)
    logger.info("PRODUCTION PREDICTIONS - GBSA MODEL")
    logger.info("="*70)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Paths
    repo_root = Path(__file__).parent.parent
    input_file = repo_root / "data/inputs/production/survival_inputs_complete.pkl"

    # Find latest GBSA model
    models_dir = repo_root / "data/outputs/production/models"
    gbsa_models = sorted(models_dir.glob("*gbsa*.joblib_*"))

    if not gbsa_models:
        logger.error("No GBSA models found!")
        return 1

    # Use most recent model
    model_path = gbsa_models[-1]
    logger.info(f"\nUsing model: {model_path.name}")
    logger.info(f"Model size: {model_path.stat().st_size / 1024:.1f} KB")

    # Load model
    logger.info("\nLoading GBSA model...")
    loaded_model = joblib.load(model_path)

    # Check if it's a pipeline or just a model
    if hasattr(loaded_model, 'named_steps'):
        # It's already a pipeline
        pipeline = loaded_model
        logger.info("✅ Model loaded as Pipeline")
    else:
        # It's just a model - need to create pipeline
        logger.info("✅ Model loaded as GBSAWrapper - creating pipeline...")
        preprocessor = make_preprocessor()

        # Fit preprocessor on production data
        logger.info("  Fitting preprocessor on production data...")
        df_prep = load_data(str(input_file), run_type="production")
        X_prep, _, _ = split_X_y(df_prep, dropna=False)
        preprocessor.fit(X_prep)
        logger.info("  ✓ Preprocessor fitted")

        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', loaded_model)
        ])
        logger.info("  ✓ Pipeline created")

    logger.info("✅ Ready for predictions")

    # Load production data
    logger.info(f"\nLoading data from {input_file}...")
    df = load_data(str(input_file), run_type="production")
    X, y, ids = split_X_y(df, dropna=False)
    logger.info(f"Loaded {len(df):,} records with {X.shape[1]} features")

    # Time horizons for predictions
    time_horizons = [3, 6, 12, 18, 24, 36]
    logger.info(f"\nPrediction time horizons: {time_horizons} months")

    # Calculate expected survival time
    logger.info("\nCalculating expected survival time (RMST)...")
    expected_months = calculate_expected_survival_time(pipeline, X)
    logger.info(f"  Mean expected survival: {expected_months.mean():.2f} months")
    logger.info(f"  Median expected survival: {np.median(expected_months):.2f} months")
    logger.info(f"  Range: [{expected_months.min():.2f}, {expected_months.max():.2f}]")

    # Predict survival functions at all time horizons
    logger.info(f"\nGenerating survival probabilities at {len(time_horizons)} time points...")
    surv_funcs = pipeline.predict_survival_function(X, times=time_horizons)
    logger.info(f"  Survival function shape: {surv_funcs.shape}")

    # Build results DataFrame
    results = {
        "account_entities_key": ids.values,
        "model_name": "gbsa",
        "expected_survival_months": expected_months
    }

    # Add survival probabilities for each time horizon
    for i, t in enumerate(time_horizons):
        col_name = f"survival_prob_{t}m"
        results[col_name] = surv_funcs[:, i]
        mean_prob = surv_funcs[:, i].mean()
        logger.info(f"  {col_name}: mean={mean_prob:.4f}")

    pred_df = pd.DataFrame(results)

    # Save predictions
    paths = get_output_paths("production")
    output_filename = f"survival_predictions_production_gbsa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    output_path = Path(paths["predictions"]) / output_filename

    # Ensure predictions directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Predictions saved to: {output_path}")
    logger.info(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"   Records: {len(pred_df):,}")
    logger.info(f"   Columns: {list(pred_df.columns)}")

    # Summary statistics
    logger.info("\n" + "="*70)
    logger.info("PREDICTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Model: GBSA (Gradient Boosting Survival Analysis)")
    logger.info(f"Input: {len(pred_df):,} accounts")
    logger.info(f"Output: {len(time_horizons)} time horizons ({min(time_horizons)}-{max(time_horizons)} months)")
    logger.info(f"Expected survival time:")
    logger.info(f"  - Mean: {expected_months.mean():.2f} months")
    logger.info(f"  - Median: {np.median(expected_months):.2f} months")
    logger.info(f"  - Std: {expected_months.std():.2f} months")

    # Risk stratification
    low_risk = (expected_months > np.percentile(expected_months, 75)).sum()
    med_risk = ((expected_months > np.percentile(expected_months, 25)) &
                (expected_months <= np.percentile(expected_months, 75))).sum()
    high_risk = (expected_months <= np.percentile(expected_months, 25)).sum()

    logger.info(f"\nRisk stratification (by expected survival quartiles):")
    logger.info(f"  - Low risk (>75th percentile): {low_risk:,} accounts ({low_risk/len(pred_df)*100:.1f}%)")
    logger.info(f"  - Medium risk (25-75th): {med_risk:,} accounts ({med_risk/len(pred_df)*100:.1f}%)")
    logger.info(f"  - High risk (<25th percentile): {high_risk:,} accounts ({high_risk/len(pred_df)*100:.1f}%)")

    logger.info("\n" + "="*70)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("="*70)
    logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
