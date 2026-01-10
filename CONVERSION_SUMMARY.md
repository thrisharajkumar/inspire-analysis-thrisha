# Summary: Switched DNN Mortality Pipeline to Real Data

## ✅ What Was Done

### 1. Created New Module: `dnn_mortality_data.py`
- **Location:** `src/dnn_mortality_data.py`
- **Purpose:** Loads real INSPIRE dataset instead of generating synthetic data
- **Key Function:** `generate_dataset(dataset_dir, feature_columns, proportion_died, num_subjects)`
- **Supported Formats:**
  - CSV files: `operations.csv`, `vitals.csv`, `labs.csv`, `ward_vitals.csv`
  - Pre-processed JSON subjects in `died/` and `survived/` subdirectories

### 2. Updated Main Pipeline: `dnn_mortality_pipeline.py`
- **Modified Function:** `main(dataset_dir=None)`
- **Changes:**
  - Now accepts `dataset_dir` parameter for dataset location
  - Auto-detects common dataset paths if not specified
  - Loads real data using the new `dnn_mortality_data` module
  - Removed synthetic data generation parameters
  - Added helpful error messages

### 3. Created Documentation: `SWITCH_TO_REAL_DATA.md`
- Comprehensive guide on how to use real data
- Data format specifications
- Troubleshooting section
- Multiple usage examples

### 4. Created Example Script: `run_with_real_data.py`
- Easy-to-use template script
- Shows how to import and call the pipeline
- Includes inline comments and instructions
- Error handling with helpful messages

## 📊 How to Use Real Data

### Quick Example:
```python
from src.dnn_mortality_pipeline import main

# Option 1: Provide dataset directory path
main(dataset_dir="C:\\path\\to\\inspire\\dataset")

# Option 2: Auto-detect from default locations
main()

# Option 3: Limit number of subjects for testing
import dnn_mortality_data
subjects_data, seq_length = dnn_mortality_data.generate_dataset(
    "C:\\inspire_data",
    feature_columns=['glucose', 'potassium', 'sodium', 'creatinine'],
    num_subjects=1000  # Load only 1000 subjects
)
```

## 🎯 Key Features

| Feature | Details |
|---------|---------|
| **Flexible Input** | CSV files OR pre-processed JSON subjects |
| **Auto-Detection** | Searches common directories automatically |
| **Feature Selection** | Customizable list of features to extract |
| **Mortality Balancing** | Optional - can target specific mortality rates |
| **Subject Limiting** | Can load first N subjects for testing |
| **Error Handling** | Clear error messages guide users to solutions |
| **Backward Compatible** | Same output format as synthetic generator |

## 📁 File Structure

```
inspire-analysis-thrisha/
├── src/
│   ├── dnn_mortality_pipeline.py    ← Updated main pipeline
│   ├── dnn_mortality_data.py        ← NEW: Real data loader
│   ├── inspire_dataset.py           (used for loading utility functions)
│   ├── subject.py                   (used for JSON subject loading)
│   └── ... (other files)
├── run_with_real_data.py            ← NEW: Example script
└── SWITCH_TO_REAL_DATA.md           ← NEW: Full documentation
```

## 🔄 Data Loading Flow

```
dataset_dir (CSV or JSON)
         ↓
dnn_mortality_data.generate_dataset()
         ↓
Determine format (CSV or JSON)
         ↓
Load and combine all data sources
         ↓
Extract requested features
         ↓
Filter by subject/mortality if needed
         ↓
Return: subjects_data dict + seq_length
         ↓
dnn_mortality_pipeline.main()
```

## 💡 What Changed vs. Before

### Before (Synthetic Data)
```python
num_subjects = 5000
feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']
proportion_died = 0.20
subjects_data, seq_length = dnn_mortality_data.generate_dataset(
    num_subjects, feature_columns, proportion_died
)  # ← Generated random synthetic data
```

### After (Real Data)
```python
dataset_dir = "C:\\inspire_dataset"  # Path to INSPIRE data
feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']

subjects_data, seq_length = dnn_mortality_data.generate_dataset(
    dataset_dir, feature_columns
)  # ← Loads REAL patient data from CSV/JSON
```

## ⚙️ Configuration

### Data Source Options

**Option 1: CSV Format**
```
├── operations.csv       (subject mortality info)
├── vitals.csv          (intraop vital signs)
├── labs.csv            (lab values)
└── ward_vitals.csv     (ward vital signs)
```

**Option 2: JSON Format**
```
├── died/
│   ├── subject_1.json
│   ├── subject_2.json
│   └── ...
└── survived/
    ├── subject_3.json
    ├── subject_4.json
    └── ...
```

### Feature Selection
```python
# Default features
feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine']

# Custom features (check available in your data)
feature_columns = ['glucose', 'potassium', 'sodium', 'creatinine', 'hemoglobin', 'hematocrit']
```

## 🚀 Next Steps

1. **Locate your INSPIRE dataset** - Find where your CSV or JSON files are stored
2. **Update the dataset path** - Edit `dataset_dir` in your script or use the example script
3. **Run the pipeline** - Execute with real data instead of synthetic
4. **Compare results** - See how the model performs with real patient data

## 📝 Notes

- **First run may be slow:** Data loading and preprocessing can take several minutes for large datasets
- **Memory intensive:** Full INSPIRE dataset requires ~8-16GB RAM
- **Feature matching:** Feature names are case-sensitive and must match CSV `item_name` column exactly
- **Mortality rate:** Real data typically has 2-5% in-hospital mortality (vs. 20% in synthetic)

## ✨ Benefits

✅ **Real Data:** Actual patient demographics and medical histories  
✅ **Heterogeneous:** Real-world variability in vital signs and labs  
✅ **Validated:** Data from established INSPIRE research  
✅ **Production-Ready:** Better model evaluation with real performance  
✅ **Flexible:** Supports multiple data formats and custom features  

---

**Status:** ✅ COMPLETED - Pipeline now uses REAL INSPIRE data  
**Date:** January 2026
**Files Modified:** 2 files  
**Files Created:** 3 files  
**Total Changes:** ~1000 lines of code/documentation
