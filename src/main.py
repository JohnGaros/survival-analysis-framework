"""Main entry point for training survival models.

Trains all survival models using the sample dataset and logs results to MLflow.
After training, generates survival probability predictions for all records.
"""
from survival_framework.train import train_all_models
from survival_framework.predict import generate_predictions
import os

if __name__ == "__main__":
    # Use relative path to sample data
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'sample', 'survival_inputs_sample2000.csv'
    )

    # Train all models
    train_all_models(csv_path)

    # Generate predictions using best model
    print("\n=== Generating Predictions ===")
    pred_path = generate_predictions(
        csv_path=csv_path,
        output_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'predictions'),
        artifacts_dir="artifacts",
        models_dir="models"
    )
    print(f"\nPredictions complete: {pred_path}")