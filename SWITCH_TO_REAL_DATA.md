# Switching from Generated to Real Data in DNN Mortality Pipeline

## Overview
The DNN mortality prediction pipeline has been updated to load **real INSPIRE dataset** instead of synthetic generated data. The new `dnn_mortality_data.py` module handles loading real data seamlessly.

## Quick Start

### Running with Real Data

**Option 1: Using CSV files (INSPIRE dataset format)**
```python
from src.dnn_mortality_pipeline import main

# Path to directory containing INSPIRE CSV files
dataset_dir = "C:\\path\\to\\inspire\\dataset"  # Contains: operations.csv, vitals.csv, labs.csv, ward_vitals.csv

main(dataset_dir=dataset_dir)
```

**Option 2: Using pre-processed JSON subjects**
```python
from src.dnn_mortality_pipeline import main

# Path to directory containing 'died' and 'survived' subdirectories with JSON files
dataset_dir = "C:\\path\\to\\inspire\\subjects"  # Contains: died/, survived/

main(dataset_dir=dataset_dir)
```

**Option 3: Automatic detection (from src/ directory)**
If you have the dataset in default locations, just call:
```python
from src.dnn_mortality_pipeline import main

main()  # Automatically looks for ../inspire_dataset or ../inspire_subjects
```

## What Changed

### New Module: `dnn_mortality_data.py`

This module provides a single function: `generate_dataset()` which:
- **Loads real INSPIRE data** from CSV files or pre-processed JSON subjects
- **Extracts specified features** (glucose, potassium, sodium, creatinine by default)
- **Handles multiple data formats** automatically
- **Maintains the same output format** as the old synthetic generator for backward compatibility
- **Balances mortality if needed** (optional parameter to achieve target mortality rate)

### Updated: `dnn_mortality_pipeline.py`

The `main()` function now:
- Accepts optional `dataset_dir` parameter
- Auto-detects common dataset locations if not specified
- Loads real data instead of generating synthetic data
- Provides helpful error messages if data path is incorrect

## Data Format Requirements

### CSV Format (Option 1)
Your dataset directory should contain these files:

```
inspire_dataset/
├── operations.csv          # Subject operations with mortality info
├── vitals.csv             # Intraoperative vital signs
├── labs.csv               # Lab values
└── ward_vitals.csv        # Ward vital signs
```

**Required CSV columns:**
- `operations.csv`: subject_id, inhosp_death_time (NULL if survived), age, sex, etc.
- `vitals.csv`: subject_id, op_id, chart_time, item_name, value
- `labs.csv`: subject_id, chart_time, item_name, value
- `ward_vitals.csv`: subject_id, chart_time, item_name, value

### JSON Subject Format (Option 2)
Your dataset directory should have structure:

```
inspire_subjects/
├── died/           # Subjects with inhosp_death_time
│   ├── subject_123456.json
│   ├── subject_123457.json
│   └── ...
└── survived/       # Subjects who survived
    ├── subject_654321.json
    ├── subject_654322.json
    └── ...
```

## Feature Columns

By default, the pipeline extracts these features:
- `glucose`
- `potassium`
- `sodium`
- `creatinine`

These are flexible and can be modified by passing different `feature_columns` to `generate_dataset()`.

## Example: Custom Usage

```python
import dnn_mortality_data

# Load real data with custom features
dataset_dir = "C:\\inspire_data"
feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine', 'hemoglobin']
proportion_died = None  # Use actual proportions from data
num_subjects = 1000     # Limit to first 1000 subjects (None = all)

subjects_data, seq_length = dnn_mortality_data.generate_dataset(
    dataset_dir, 
    feature_columns, 
    proportion_died, 
    num_subjects
)

print(f"Loaded {len(subjects_data)} subjects")
print(f"Sequence length: {seq_length}")
```

## Key Differences from Synthetic Data

| Aspect | Synthetic | Real |
|--------|-----------|------|
| Data Source | Random generation | INSPIRE CSV/JSON files |
| Mortality Rate | Fixed (0.20 or custom) | Actual data (typically ~2-5%) |
| Feature Distributions | Uniform/Normal random | Real medical data |
| Subject Variety | Homogeneous | Real patient heterogeneity |
| Chart Times | Synthetic sequence | Real clinical timestamps |
| Reproducibility | Deterministic | Same across runs if data unchanged |

## Troubleshooting

### Error: "No dataset directory found"
**Solution:** Provide the full path to your dataset:
```python
main(dataset_dir="C:\\Users\\YourName\\Documents\\inspire_dataset")
```

### Error: "ValueError: Not enough subjects to achieve desired not_survived_pct"
**Solution:** The dataset doesn't have enough subjects with the desired mortality proportion. Use:
```python
# Use actual proportions from data instead of forcing a specific proportion
main(dataset_dir=dataset_dir)  # proportion_died is automatically None internally
```

### Error: "No feature data extracted for any subjects"
**Solution:** The feature names don't match what's in your data. Check available features:
```python
# Check what features are in your CSV files
import pandas as pd
vitals = pd.read_csv("path/to/vitals.csv")
print(vitals['item_name'].unique())  # See available feature names
```

## Performance Notes

- **CSV loading:** ~20-100 seconds for full INSPIRE dataset (66M+ vitals records)
- **JSON loading:** ~30-60 seconds for pre-processed subjects
- **Memory usage:** 4-16 GB depending on number of subjects loaded
- **First run may be slow:** Data alignment and normalization are performed

## Next Steps

1. **Locate your INSPIRE dataset** (CSV files or JSON subjects)
2. **Update the dataset_dir path** in your code
3. **Run the pipeline** with real data
4. **Compare results** with synthetic data baseline if available

## Support

For issues or questions:
- Check that all required CSV files are present
- Verify that feature names match exactly (case-sensitive)
- Ensure chart_time values are numeric
- Look for error messages that indicate missing data

---

**Modified:** January 2026
**Module:** `dnn_mortality_data.py`
**Status:** Real data loading enabled ✓
