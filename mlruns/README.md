# mlruns/

## Purpose

MLflow experiment tracking directory (auto-generated).

## Contents

MLflow stores experiment metadata, metrics, parameters, and artifacts in this directory:
- `0/` - Default experiment
- `{experiment_id}/` - Experiment-specific runs
- Each run has subdirectories: `metrics/`, `params/`, `tags/`, `artifacts/`

## Usage

```bash
# View experiments in MLflow UI
mlflow ui

# Access at http://localhost:5000
```

## Git Status

**Gitignored** - MLflow tracking data is environment-specific and regenerated per run.

## Cleanup

```bash
# Remove all tracking data
rm -rf mlruns/

# Will be recreated automatically on next run if tracking is enabled
```

**Note:** If experiment tracking is not needed, MLflow usage can be disabled in the training code.
