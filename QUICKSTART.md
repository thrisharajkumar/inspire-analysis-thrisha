# Quick Start: Using Real Data with DNN Mortality Pipeline

## 30-Second Setup

### Step 1: Find Your Data
You need **one** of these:
- A folder with: `operations.csv`, `vitals.csv`, `labs.csv`, `ward_vitals.csv` (CSV format)
- A folder with: `died/` and `survived/` subdirectories containing JSON files (JSON format)

### Step 2: Update Path
Edit this in your Python script or in `run_with_real_data.py`:
```python
dataset_dir = r"C:\path\to\your\inspire\data"  # Update this!
```

### Step 3: Run It
```python
from src.dnn_mortality_pipeline import main

main(dataset_dir=r"C:\path\to\your\inspire\data")
```

That's it! 🎉

---

## What Just Happened

✅ **Old way:** Generated random synthetic data (5000 subjects, 20% mortality)  
✅ **New way:** Loads REAL patient data from your INSPIRE dataset  

## File Reference

| File | Purpose |
|------|---------|
| `src/dnn_mortality_data.py` | New module that loads real data |
| `src/dnn_mortality_pipeline.py` | Updated to use real data instead of synthetic |
| `SWITCH_TO_REAL_DATA.md` | Complete documentation |
| `run_with_real_data.py` | Ready-to-use example script |
| `CONVERSION_SUMMARY.md` | Technical summary of changes |

## Common Tasks

### Load Real Data & Use It
```python
from src.dnn_mortality_pipeline import main

main(dataset_dir="C:\\inspire_dataset")
```

### Load Fewer Subjects (for testing)
```python
import sys
sys.path.insert(0, 'src')
import dnn_mortality_data

subjects_data, seq_length = dnn_mortality_data.generate_dataset(
    "C:\\inspire_dataset",
    feature_columns=['glucose', 'potassium', 'sodium', 'creatinine'],
    num_subjects=100  # Just load 100 subjects for testing
)
```

### Use Different Features
```python
dnn_mortality_data.generate_dataset(
    "C:\\inspire_dataset",
    feature_columns=['glucose', 'hemoglobin', 'platelet']  # Your custom features
)
```

### Auto-Detect Dataset Location
```python
# If you put data in ../inspire_dataset or ../inspire_subjects
main()  # Finds it automatically!
```

## ❌ Common Errors & Fixes

| Error | Fix |
|-------|-----|
| "No dataset directory found" | Provide full path: `main(dataset_dir="C:\\...")` |
| "operations.csv not found" | Check file names are exactly correct (case-sensitive on Linux/Mac) |
| "No feature data extracted" | Feature names must match CSV `item_name` column exactly |
| "Not enough subjects" | Use actual data proportions: `proportion_died=None` |

## 📊 What You Get

The function returns a `subjects_data` dictionary with:
- **Keys:** Subject IDs
- **Values:** Dict with:
  - `label`: 0 (survived) or 1 (died in hospital)
  - `timeseries`: Dict of feature DataFrames with chart_time
  - `subject_id`: The subject identifier

Plus `seq_length`: Maximum time range across all subjects

## 🔗 More Info

- See **SWITCH_TO_REAL_DATA.md** for full documentation
- See **run_with_real_data.py** for complete example
- See **CONVERSION_SUMMARY.md** for technical details

---

**Ready to go!** Update your dataset path and run! 🚀
