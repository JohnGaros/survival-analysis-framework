# Data Directory

This directory contains input datasets for the survival modeling framework.

## Structure

- `sample/` - Sample datasets for testing and development

## Sample Data

### survival_inputs_sample2000.csv

A sample dataset with 2000 customer records for testing the survival framework.

**Required Columns:**
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

## Adding Your Own Data

To use your own dataset:

1. Ensure your CSV contains all required columns listed above
2. Place your CSV file in `data/` or a subdirectory
3. Run training with: `python -m survival_framework.train --csv path/to/your/data.csv`

Or update `src/main.py` to point to your dataset.
