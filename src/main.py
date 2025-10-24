"""Main entry point for training survival models.

Trains all survival models and generates predictions.
Supports both sample (development) and production (full data) runs.
Accepts CSV or pickle input formats.
Supports parallel execution for faster training on large datasets.

Can be used as CLI or imported as a function.
"""
from survival_framework.train import train_all_models
from survival_framework.predict import generate_predictions
from survival_framework.data import RunType
from survival_framework.config import (
    ExecutionConfig,
    ExecutionMode,
    create_execution_config,
    get_data_size_mb,
    select_execution_mode
)
import os
import argparse
from typing import Optional, Literal


def run_pipeline(
    input_file: str = "data/inputs/sample/survival_inputs_sample2000.csv",
    run_type: RunType = "sample",
    predict_only: bool = False,
    train_only: bool = False,
    execution_config: Optional[ExecutionConfig] = None,
) -> int:
    """Run the survival analysis pipeline.

    This function can be called directly from Python code or via CLI.
    Trains models and/or generates predictions based on the specified mode.
    Supports parallel execution for faster training on large datasets.

    Args:
        input_file: Path to input file (CSV or pickle). Can be relative or absolute.
            Default: "data/inputs/sample/survival_inputs_sample2000.csv"
        run_type: Type of run - "sample" for development, "production" for full data.
            Default: "sample"
        predict_only: If True, skip training and only generate predictions using
            existing models. Default: False
        train_only: If True, only train models and skip prediction generation.
            Default: False
        execution_config: Configuration for execution mode and parallelization.
            If None, creates default config with auto-detection based on data size.

    Returns:
        Exit code (0 for success, 1 for failure)

    Example:
        >>> # Direct function call
        >>> from main import run_pipeline
        >>> run_pipeline(
        ...     input_file="data/inputs/sample/data.csv",
        ...     run_type="sample",
        ...     predict_only=False,
        ...     train_only=False
        ... )
        0

        >>> # Production run
        >>> run_pipeline(
        ...     input_file="data/inputs/production/survival_inputs_complete.pkl",
        ...     run_type="production"
        ... )
        0

        >>> # Prediction only
        >>> run_pipeline(
        ...     input_file="data/inputs/sample/data.csv",
        ...     run_type="sample",
        ...     predict_only=True
        ... )
        0

    Notes:
        - If both predict_only and train_only are True, both phases run (default behavior)
        - Relative paths are resolved relative to the repository root
        - Input file existence is validated before execution
    """
    # Resolve relative paths
    if not os.path.isabs(input_file):
        # Get repository root (parent of src/)
        repo_root = os.path.dirname(os.path.dirname(__file__))
        input_file = os.path.join(repo_root, input_file)

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return 1

    # Create or validate execution config
    if execution_config is None:
        # Auto-detect execution mode based on data size
        data_size_mb = get_data_size_mb(input_file)
        auto_mode = select_execution_mode(data_size_mb)
        execution_config = ExecutionConfig(mode=auto_mode, n_jobs=-1, auto_mode=True)

    # If both flags are True, run both phases (ignore flags)
    if predict_only and train_only:
        predict_only = False
        train_only = False

    # Print run configuration
    print("\n" + "=" * 70)
    print(f"SURVIVAL ANALYSIS FRAMEWORK - {run_type.upper()} RUN")
    print("=" * 70)
    print(f"Input file: {input_file}")
    print(f"File size:  {get_data_size_mb(input_file):.1f} MB")
    print(f"Run type:   {run_type}")
    print(f"Mode:       {'Predict only' if predict_only else 'Train only' if train_only else 'Train + Predict'}")
    print(f"Execution:  {execution_config}")
    print("=" * 70 + "\n")

    # Training phase
    if not predict_only:
        print("\n" + "=" * 70)
        print("PHASE 1: MODEL TRAINING")
        print("=" * 70)
        train_all_models(input_file, run_type=run_type, execution_config=execution_config)

    # Prediction phase
    if not train_only:
        print("\n" + "=" * 70)
        print("PHASE 2: PREDICTION GENERATION")
        print("=" * 70)
        pred_path = generate_predictions(input_file, run_type=run_type)
        print(f"\n✓ Predictions complete: {pred_path}")

    print("\n" + "=" * 70)
    print(f"✓ {run_type.upper()} RUN COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return 0


def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Survival Analysis Framework - Train models and generate predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sample run with CSV (auto-detect execution mode)
  python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample

  # Production run with pickle (auto-enables multiprocessing for large data)
  python src/main.py --input data/inputs/production/all_customers.pkl --run-type production

  # Force multiprocessing with 8 cores
  python src/main.py --input data.pkl --run-type production --execution-mode mp --n-jobs 8

  # Silent execution (no progress bars)
  python src/main.py --input data.pkl --run-type production --verbose 0

  # Skip training, only predict
  python src/main.py --input data/inputs/sample/data.csv --run-type sample --predict-only
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/inputs/sample/survival_inputs_sample2000.csv",
        help="Path to input file (CSV or pickle). Default: sample CSV"
    )

    parser.add_argument(
        "--run-type",
        type=str,
        choices=["sample", "production"],
        default="sample",
        help="Run type: 'sample' for development, 'production' for full data. Default: sample"
    )

    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training and only generate predictions using existing models"
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models, skip prediction generation"
    )

    parser.add_argument(
        "--execution-mode",
        type=str,
        choices=["pandas", "mp", "spark_local", "spark_cluster"],
        default=None,
        help="Execution mode: 'pandas' (single-thread), 'mp' (multiprocessing), 'spark_local' (PySpark local), 'spark_cluster' (PySpark cluster). Default: auto-detect based on data size"
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for multiprocessing. -1 means use all cores. Default: -1"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=10,
        choices=[0, 10, 50],
        help="Verbosity level: 0 (silent), 10 (progress bars), 50 (detailed). Default: 10"
    )

    args = parser.parse_args()

    # Create execution config from CLI arguments
    if args.execution_mode is None:
        # Auto-detect mode
        execution_config = None  # Will be auto-detected in run_pipeline
    else:
        execution_config = create_execution_config(
            mode=args.execution_mode,
            n_jobs=args.n_jobs,
            auto_mode=False,
            verbose=args.verbose
        )

    # Call the main pipeline function with parsed arguments
    return run_pipeline(
        input_file=args.input,
        run_type=args.run_type,
        predict_only=args.predict_only,
        train_only=args.train_only,
        execution_config=execution_config
    )


if __name__ == "__main__":
    exit(main())