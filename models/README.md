# models/

## Purpose

Trained survival models serialized for later use.

## Contents

Serialized model files (`.joblib` format) for each trained survival model:
- `cox_ph.joblib`
- `coxnet.joblib`
- `weibull_aft.joblib`
- `gbsa.joblib`
- `rsf.joblib`

## Generation

Models are saved automatically during training:
```python
from survival_framework.train import train_all_models
train_all_models('data/sample/survival_inputs_sample2000.csv')
```

## Git Status

**Gitignored** - Models are large binary files that can be regenerated.

## Loading Models

```python
import joblib

# Load saved model
model = joblib.load('models/cox_ph.joblib')

# Make predictions
risk_scores = model.predict(X_new)
surv_funcs = model.predict_survival_function(X_new)
```

## Cleanup

```bash
rm -rf models/  # Remove all saved models
python src/main.py  # Regenerate
```
