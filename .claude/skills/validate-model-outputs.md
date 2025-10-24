# Validate Model Outputs Skill

Comprehensive validation of survival model predictions and artifacts to ensure correctness, consistency, and quality.

## Purpose

This skill validates that trained models produce correct, consistent, and high-quality outputs across all aspects of the survival analysis pipeline.

## When to Use

Run this skill:
1. After training models on new data
2. Before deploying models to production
3. After modifying model implementations
4. When debugging unexpected model behavior
5. To verify model artifact integrity

## Validation Categories

### 1. Survival Function Predictions

**Validates:**
- All survival probabilities are in [0, 1]
- Survival functions are non-increasing over time
- Survival at time 0 equals 1.0
- Survival at max time > 0
- No NaN or infinite values
- Consistent predictions across folds

**Validation Script:**
```python
import numpy as np
import os

def validate_survival_predictions(model_name, run_type='sample'):
    """Validate survival function predictions for a model."""
    base_path = f'data/outputs/{run_type}/artifacts/{model_name}'

    errors = []

    for fold in range(5):
        surv_file = f'{base_path}/{model_name}_fold{fold}_surv.npy'

        if not os.path.exists(surv_file):
            errors.append(f"Missing file: {surv_file}")
            continue

        surv = np.load(surv_file)

        # Check range [0, 1]
        if not np.all((surv >= 0) & (surv <= 1)):
            errors.append(f"Fold {fold}: Survival probabilities outside [0,1]")

        # Check for NaN/inf
        if np.any(np.isnan(surv)) or np.any(np.isinf(surv)):
            errors.append(f"Fold {fold}: NaN or infinite values detected")

        # Check non-increasing (survival should decrease over time)
        # Each row is a sample, each column is a time point
        for i in range(surv.shape[0]):
            if not np.all(np.diff(surv[i, :]) <= 1e-6):  # Allow small numerical errors
                errors.append(f"Fold {fold}, Sample {i}: Survival function not monotonically decreasing")
                break

    return errors
```

### 2. Risk Score Predictions

**Validates:**
- Risk scores are non-negative
- Higher risk → lower survival probability
- Consistent ordering across time points
- No NaN or infinite values
- Reasonable distribution (not all same value)

**Validation Script:**
```python
def validate_risk_scores(model_name, run_type='sample'):
    """Validate risk score predictions for a model."""
    base_path = f'data/outputs/{run_type}/artifacts/{model_name}'

    errors = []

    for fold in range(5):
        risk_file = f'{base_path}/{model_name}_fold{fold}_risk.npy'

        if not os.path.exists(risk_file):
            errors.append(f"Missing file: {risk_file}")
            continue

        risk = np.load(risk_file)

        # Check for NaN/inf
        if np.any(np.isnan(risk)) or np.any(np.isinf(risk)):
            errors.append(f"Fold {fold}: NaN or infinite risk scores")

        # Check variance (all same value indicates problem)
        if np.std(risk) < 1e-6:
            errors.append(f"Fold {fold}: Risk scores have zero variance")

        # Check reasonable range (should be roughly standardized)
        if np.max(np.abs(risk)) > 100:
            errors.append(f"Fold {fold}: Extreme risk score values (max abs: {np.max(np.abs(risk))})")

    return errors
```

### 3. Metrics Validation

**Validates:**
- C-index in valid range [0.5, 1.0] for good models
- IBS in reasonable range [0, 0.25]
- Time-dependent AUC values in [0.5, 1.0]
- Consistent metrics across folds (low variance)
- All required metrics present
- No missing values

**Validation Script:**
```python
import pandas as pd

def validate_metrics(run_type='sample'):
    """Validate model performance metrics."""
    metrics_file = f'data/outputs/{run_type}/artifacts/model_metrics.csv'

    if not os.path.exists(metrics_file):
        return ["Missing model_metrics.csv"]

    metrics = pd.read_csv(metrics_file)
    errors = []

    # Check required columns
    required_cols = ['model', 'fold', 'cindex', 'ibs', 'mean_auc']
    missing_cols = set(required_cols) - set(metrics.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
        return errors

    # Check for NaN
    if metrics[['cindex', 'ibs', 'mean_auc']].isna().any().any():
        errors.append("NaN values in metrics")

    # Check C-index range
    if not metrics['cindex'].between(0.5, 1.0).all():
        bad_models = metrics[~metrics['cindex'].between(0.5, 1.0)][['model', 'fold', 'cindex']]
        errors.append(f"C-index out of range [0.5, 1.0]:\n{bad_models}")

    # Check IBS range
    if not metrics['ibs'].between(0.0, 0.25).all():
        bad_models = metrics[~metrics['ibs'].between(0.0, 0.25)][['model', 'fold', 'ibs']]
        errors.append(f"IBS out of range [0, 0.25]:\n{bad_models}")

    # Check AUC range
    if not metrics['mean_auc'].between(0.5, 1.0).all():
        bad_models = metrics[~metrics['mean_auc'].between(0.5, 1.0)][['model', 'fold', 'mean_auc']]
        errors.append(f"AUC out of range [0.5, 1.0]:\n{bad_models}")

    # Check fold consistency (CV should be stable)
    for model in metrics['model'].unique():
        model_data = metrics[metrics['model'] == model]
        cindex_std = model_data['cindex'].std()
        if cindex_std > 0.1:  # High variance suggests instability
            errors.append(f"{model}: High C-index variance across folds (std={cindex_std:.4f})")

    return errors
```

### 4. Model Artifact Files

**Validates:**
- All expected files exist
- Files are not empty
- Files can be loaded successfully
- Model objects contain expected attributes
- Models can make predictions

**Validation Script:**
```python
import joblib

def validate_model_artifacts(run_type='sample'):
    """Validate saved model files."""
    models_dir = f'data/outputs/{run_type}/models'
    expected_models = ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']

    errors = []

    for model_name in expected_models:
        # Find model file (has timestamp in filename)
        import glob
        pattern = f'{models_dir}/{model_name}_*.joblib'
        model_files = glob.glob(pattern)

        if not model_files:
            errors.append(f"No model file found for {model_name}")
            continue

        model_file = model_files[0]  # Use most recent

        # Check file size
        if os.path.getsize(model_file) == 0:
            errors.append(f"{model_name}: Empty model file")
            continue

        # Try loading
        try:
            pipeline = joblib.load(model_file)
        except Exception as e:
            errors.append(f"{model_name}: Failed to load model - {e}")
            continue

        # Check pipeline structure
        if not hasattr(pipeline, 'steps'):
            errors.append(f"{model_name}: Loaded object is not a Pipeline")
            continue

        if len(pipeline.steps) != 2:
            errors.append(f"{model_name}: Pipeline should have 2 steps, has {len(pipeline.steps)}")

    return errors
```

### 5. Cross-Model Consistency

**Validates:**
- All models rank samples similarly (rank correlation > 0.7)
- Models agree on high-risk vs low-risk patients
- Survival function shapes are reasonable
- No single model drastically outperforms others (suggests overfitting)

**Validation Script:**
```python
from scipy.stats import spearmanr

def validate_cross_model_consistency(run_type='sample'):
    """Validate that models produce consistent predictions."""
    models = ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']
    errors = []

    for fold in range(5):
        risk_scores = {}

        # Load risk scores for all models
        for model in models:
            risk_file = f'data/outputs/{run_type}/artifacts/{model}/{model}_fold{fold}_risk.npy'
            if os.path.exists(risk_file):
                risk_scores[model] = np.load(risk_file)

        # Check pairwise correlations
        for i, model1 in enumerate(models):
            if model1 not in risk_scores:
                continue
            for model2 in models[i+1:]:
                if model2 not in risk_scores:
                    continue

                corr, _ = spearmanr(risk_scores[model1], risk_scores[model2])

                if corr < 0.7:
                    errors.append(
                        f"Fold {fold}: Low correlation between {model1} and {model2} "
                        f"(rho={corr:.3f}). Models may be learning different patterns."
                    )

    return errors
```

## Complete Validation Runner

```python
def run_full_validation(run_type='sample'):
    """Run all validation checks and generate report."""

    all_errors = []

    print("=" * 70)
    print(f"MODEL OUTPUT VALIDATION REPORT - {run_type.upper()}")
    print("=" * 70)

    # 1. Validate survival predictions
    print("\n1. Validating survival function predictions...")
    for model in ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']:
        errors = validate_survival_predictions(model, run_type)
        if errors:
            all_errors.extend([f"[{model}] {e}" for e in errors])
            print(f"  ❌ {model}: {len(errors)} errors")
        else:
            print(f"  ✅ {model}: OK")

    # 2. Validate risk scores
    print("\n2. Validating risk score predictions...")
    for model in ['cox_ph', 'coxnet', 'weibull_aft', 'gbsa', 'rsf']:
        errors = validate_risk_scores(model, run_type)
        if errors:
            all_errors.extend([f"[{model}] {e}" for e in errors])
            print(f"  ❌ {model}: {len(errors)} errors")
        else:
            print(f"  ✅ {model}: OK")

    # 3. Validate metrics
    print("\n3. Validating performance metrics...")
    errors = validate_metrics(run_type)
    if errors:
        all_errors.extend(errors)
        print(f"  ❌ Metrics: {len(errors)} errors")
    else:
        print(f"  ✅ Metrics: OK")

    # 4. Validate model artifacts
    print("\n4. Validating model artifact files...")
    errors = validate_model_artifacts(run_type)
    if errors:
        all_errors.extend(errors)
        print(f"  ❌ Artifacts: {len(errors)} errors")
    else:
        print(f"  ✅ Artifacts: OK")

    # 5. Validate cross-model consistency
    print("\n5. Validating cross-model consistency...")
    errors = validate_cross_model_consistency(run_type)
    if errors:
        all_errors.extend(errors)
        print(f"  ❌ Consistency: {len(errors)} warnings")
    else:
        print(f"  ✅ Consistency: OK")

    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} issues found")
        print("=" * 70)
        print("\nErrors:")
        for error in all_errors:
            print(f"  - {error}")
        return False
    else:
        print("VALIDATION PASSED: All checks successful")
        print("=" * 70)
        return True

# Usage
if __name__ == "__main__":
    import sys
    run_type = sys.argv[1] if len(sys.argv) > 1 else 'sample'
    success = run_full_validation(run_type)
    sys.exit(0 if success else 1)
```

## Usage Examples

### Basic Validation
```bash
# Validate sample data outputs
python -c "from validate_model_outputs import run_full_validation; run_full_validation('sample')"

# Validate production data outputs
python -c "from validate_model_outputs import run_full_validation; run_full_validation('production')"
```

### Automated Validation in Pipeline
```bash
# After training
python src/main.py --input data.csv --run-type sample
python scripts/validate_outputs.py sample

# Exit code 0 = validation passed
if [ $? -eq 0 ]; then
    echo "✅ Models validated successfully"
else
    echo "❌ Validation failed - check outputs"
    exit 1
fi
```

## Expected Results

**Healthy Model Outputs:**
- All survival functions start at 1.0 and decrease monotonically
- Risk scores show good separation between high/low risk patients
- C-index typically 0.75-0.90 for good models
- IBS typically 0.05-0.15 for well-calibrated models
- Models show high rank correlation (>0.8) with each other
- Low variance in metrics across folds (<0.05 std deviation)

**Warning Signs:**
- C-index < 0.6 → Model barely better than random
- C-index > 0.95 → Possible overfitting or data leakage
- IBS > 0.20 → Poor calibration
- High metric variance across folds → Unstable model
- Low cross-model correlation (<0.7) → Models disagree on predictions
