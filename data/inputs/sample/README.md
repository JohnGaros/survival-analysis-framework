# data/sample/

## Purpose

Sample datasets for development, testing, and demonstrations.

## Contents

- `survival_inputs_sample2000.csv` - 2000-row sample dataset for survival modeling

## Characteristics

**Data properties:**
- Realistic edge cases (zero survival times, missing values, rare categories)
- Large enough to trigger numerical issues (LAPACK errors, etc.)
- Small enough for fast integration testing (~90s)

## Usage

```python
# In code
from survival_framework.data import load_data
df = load_data('data/sample/survival_inputs_sample2000.csv')

# In tests (via fixtures)
def test_example(sample_data_path):
    df = load_data(str(sample_data_path))
```

## Git Status

**Tracked** - Sample data is committed to version control for reproducible testing.

**Note:** Larger datasets should go in `data/raw/` (gitignored).
