# Data Directory

This directory contains input datasets and outputs for the survival modeling framework.

## Directory Structure

```
data/
├── inputs/
│   ├── sample/              # Development data (~2K records)
│   │   ├── survival_inputs_sample2000.csv
│   │   └── README.md
│   └── production/          # Full population data
│       ├── survival_inputs_complete.pkl (30 MB)
│       └── README.md
└── outputs/
    ├── sample/              # Sample run outputs
    │   ├── predictions/     # Prediction CSVs
    │   ├── artifacts/       # Training metrics, PH flags
    │   ├── models/          # Saved model files
    │   └── mlruns/          # MLflow tracking
    └── production/          # Production run outputs
        ├── predictions/
        ├── artifacts/
        ├── models/
        └── mlruns/
```

## Input Data

### Sample Data (Development)
- **Location:** `inputs/sample/`
- **Size:** ~2,000 records
- **Format:** CSV
- **Purpose:** Development, testing, rapid iteration
- **Usage:** `python src/main.py --input data/inputs/sample/survival_inputs_sample2000.csv --run-type sample`

### Production Data (Full Population)
- **Location:** `inputs/production/`
- **Size:** Full dataset (~30 MB pickle)
- **Format:** Pickle (recommended for large data) or CSV
- **Purpose:** Final production models, actual predictions
- **Usage:** `python src/main.py --input data/inputs/production/survival_inputs_complete.pkl --run-type production`

## Output Data

All outputs are automatically organized by run type:

### Sample Outputs
- **Location:** `outputs/sample/`
- **Contents:** Models, predictions, metrics from sample runs
- **Naming:** All files prefixed with `sample_`
- **Example:** `sample_cox_ph_20251023_143052.joblib`

### Production Outputs
- **Location:** `outputs/production/`
- **Contents:** Models, predictions, metrics from production runs
- **Naming:** All files prefixed with `production_`
- **Example:** `production_cox_ph_20251023_143052.joblib`

## Required Data Schema

All input files must contain:

**ID and Target:**
- `account_entities_key` - Unique customer identifier
- `survival_months` - Time to event or censoring (in months)
- `is_terminated` - Event indicator (1 = terminated, 0 = censored)

**Numeric Features (9):**
- `debit_exp_smooth` - Exponentially smoothed debit amounts
- `credit_exp_smooth` - Exponentially smoothed credit amounts
- `balance_exp_smooth` - Exponentially smoothed account balance
- `past_due_balance_exp_smooth` - Exponentially smoothed past due balance
- `oldest_past_due_exp_smooth` - Exponentially smoothed oldest past due days
- `waobd_exp_smooth` - Exponentially smoothed weighted average outstanding balance days
- `total_settlements` - Total number of settlement agreements
- `active_settlements` - Number of active settlements
- `defaulted_settlements` - Number of defaulted settlements

**Categorical Features (2):**
- `typeoftariff_coarse` - Coarse tariff type category
- `risk_level_coarse` - Coarse risk level category

## Supported File Formats

### CSV (.csv)
- Human-readable
- Good for small to medium datasets (<10MB)
- Slower to load

### Pickle (.pkl, .pickle)
- Binary format
- **Recommended for large datasets (>10MB)**
- Much faster to load
- Example: `survival_inputs_complete.pkl` (30 MB)

## Usage Examples

### Sample Run (Development)
```bash
python src/main.py \
  --input data/inputs/sample/survival_inputs_sample2000.csv \
  --run-type sample
```

### Production Run (Full Data)
```bash
python src/main.py \
  --input data/inputs/production/survival_inputs_complete.pkl \
  --run-type production
```

### Convert CSV to Pickle
```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/inputs/production/large_data.csv')

# Save as pickle (faster for repeated use)
df.to_pickle('data/inputs/production/large_data.pkl')
```

## Adding Your Own Data

1. **For development:** Place CSV in `inputs/sample/`
2. **For production:** Place CSV or pickle in `inputs/production/`
3. **Ensure schema matches** required columns above
4. **Run with appropriate `--run-type`** flag

## Benefits of This Structure

✅ **Clear Separation:** Sample and production runs never conflict
✅ **Easy Comparison:** Compare sample vs production results side-by-side
✅ **Safe Development:** Work with sample data without affecting production
✅ **Reproducibility:** Each run type maintains its own complete history
✅ **Scalability:** Production ready with pickle format support

## Notes

- Input data files are tracked in git (sample CSV, production pickle)
- Output directories are gitignored (generated artifacts)
- Always use `--run-type` flag to specify which dataset you're using
- Pickle files are faster but Python version-specific
