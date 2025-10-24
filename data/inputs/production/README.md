# Production Input Data

This directory contains full population datasets for production runs.

## Current Files

### survival_inputs_complete.pkl (30 MB)

Complete survival analysis dataset in pickle format for efficient loading.

**Format:** Pickle (.pkl)
**Size:** ~30 MB
**Usage:**
```bash
python src/main.py --input data/inputs/production/survival_inputs_complete.pkl --run-type production
```

## Data Requirements

Production input files must contain the same schema as sample data:

**Required Columns:**
- `account_entities_key` - Unique customer identifier
- `survival_months` - Time to event or censoring (in months)
- `is_terminated` - Event indicator (1 = terminated, 0 = censored)

**Numeric Features (9):**
- `debit_exp_smooth`
- `credit_exp_smooth`
- `balance_exp_smooth`
- `past_due_balance_exp_smooth`
- `oldest_past_due_exp_smooth`
- `waobd_exp_smooth`
- `total_settlements`
- `active_settlements`
- `defaulted_settlements`

**Categorical Features (2):**
- `typeoftariff_coarse`
- `risk_level_coarse`

## Supported Formats

- **CSV** (.csv) - Human-readable, slower for large files
- **Pickle** (.pkl, .pickle) - Binary format, faster for large files (recommended for production)

## Adding Production Data

1. Place your production data file in this directory
2. Ensure it follows the required schema
3. Use pickle format for files >10MB for better performance
4. Run with `--run-type production`

Example:
```bash
# Save DataFrame as pickle
import pandas as pd
df = pd.read_csv('large_data.csv')
df.to_pickle('data/inputs/production/my_production_data.pkl')

# Run production training
python src/main.py --input data/inputs/production/my_production_data.pkl --run-type production
```

## Outputs

Production runs will save all outputs to:
- `data/outputs/production/predictions/` - Prediction CSVs
- `data/outputs/production/models/` - Trained models
- `data/outputs/production/artifacts/` - Training metrics and artifacts
- `data/outputs/production/mlruns/` - MLflow experiment tracking

## Best Practices

1. **Use pickle format** for large datasets (>10MB) for faster loading
2. **Validate schema** before running production training
3. **Monitor disk space** - production runs generate large model files
4. **Keep backups** of production input data
5. **Document data sources** and preprocessing steps

## Notes

- Production data files are gitignored by default
- Only directory structure is tracked in git
- Pickle files are Python version-specific (use same Python version for saving and loading)
- For cross-platform compatibility, consider using CSV or Parquet formats
