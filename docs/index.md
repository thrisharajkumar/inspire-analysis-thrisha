# INSPIRE DNN Mortality Pipeline — Complete Project README

> This README is the single source of truth for the project. It contains everything needed to continue in a new conversation without uploading any files — the dataset structure, the code architecture, the current results, and the full research roadmap.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Aims and Objectives](#2-aims-and-objectives)
3. [Repository Structure](#3-repository-structure)
4. [Dataset Structure — INSPIRE](#4-dataset-structure-inspire)
5. [Subject JSON File — Complete Format](#5-subject-json-file-complete-format)
6. [Real Patient Example — Subject 100033460](#6-real-patient-example-subject-100033460)
7. [Dataset Subset Structure on Disk](#7-dataset-subset-structure-on-disk)
8. [Current Pipeline Architecture](#8-current-pipeline-architecture)
9. [Current Config Values](#9-current-config-values)
10. [Current Results](#10-current-results)
11. [All Available Features — What Is Used vs What Exists](#11-all-available-features-what-is-used-vs-what-exists)
12. [How to Run — Colab Workflow](#12-how-to-run-colab-workflow)
13. [Known Issues and Fixes Applied](#13-known-issues-and-fixes-applied)
14. [Research Direction and Next Steps](#14-research-direction-and-next-steps)
15. [Research Exploration Roadmap](#15-research-exploration-roadmap)
16. [Key Papers and References](#16-key-papers-and-references)

---

## 1. Project Overview

**Goal:** Predict 30-day mortality after surgery using a deep learning transformer that reads clinical time series — and explain those predictions in language a surgeon can act on (which organ system is failing, what the specific drivers are, how confident the model is).

**Dataset:** INSPIRE — a Korean single-centre perioperative dataset of ~99,886 surgical patients (2011–2020). Contains labs, ward vitals, intraoperative vitals, medications, and ICD-10 diagnoses per patient, all timestamped in minutes relative to hospital admission.

> **Note on death count:** this document states **942 deaths** throughout (consistent with the `pos_weight ≈ 105` calculation in Section 9). A different figure (938) has come up in informal notes elsewhere — worth confirming the exact number against the source data before it's used in the paper or any published figure.

**Current state:** Two-phase transformer pipeline running on real INSPIRE data. Autoencoder pretrains on unlabelled time series, classifier fine-tunes end-to-end. AUROC = 0.78 on a 29-patient real-data subset.

**Benchmark to beat:** Shickel et al. 2023 (Scientific Reports, doi:10.1038/s41598-023-27418-5) — same problem, University of Florida, 56,242 patients, AUROC = 0.92 using preoperative + intraoperative data.

**Compared models in this project:**
- NELA score (`nela.py`) — fixed clinical equation, 25 variables, no training
- GBM (`gbm_mortality.py`) — Saranya's XGBoost model with frailty analysis
- DNN transformer (`dnn_mortality_pipeline.py`) — this project's main contribution

---

## 2. Aims and Objectives

**Aim:** Build an explainable deep learning pipeline that predicts mortality from the INSPIRE dataset, organised so that predictions can be broken down by physiological system rather than returned as a single opaque score.

**Current focus (near-term priority):**
1. **Model interpretability**, tied specifically to two clinical questions rather than interpretability in the abstract:
   - **ICD-10 codes vs. mortality rate** — diagnosis codes and operation-type codes, explored separately.
   - **Surgical department vs. mortality risk** — 15 departments, analysed jointly with ICD-10 rather than independently.

**Supporting objectives:**
1. **Data coverage (EDA phase)** — audit all 126 available parameters across `labs` (38), `ward_vitals` (16), and intra-operative `vitals` (72) for missingness, measurement frequency, and outlier prevalence. *(Whether intra-op vitals are included depends on an open scope decision — see Section 4.)*
2. **Feature selection by organ system** — group usable parameters into renal, cardiovascular, respiratory, metabolic/hepatic, and haematological sets (Section 11), expanding from the current 7-feature pilot.
3. **ICD-10 deep dive** — diagnosis codes vs. operation/procedure codes, multi-code patients, overlap with the existing HFRS frailty codes (Section 15).
4. **Frailty integration** — fix the HFRS time-window gap, then evaluate HFRS as a baseline and as an added feature, stratified by scheduled vs. emergency surgery (see Section 15, item 9).
5. **Model scale-up** — move from the 29-patient development subset to the full 99,886-patient cohort.
6. **Explainability architecture** — implement system-separated encoders so mortality predictions decompose into per-organ-system contributions (Section 14).
7. **Benchmarking** — compare NELA, GBM (± frailty), transformer, GRU, and TCN on the same task.

The full, un-prioritised list of everything being explored — sampling strategies for class imbalance, missing-data handling, conditional dependencies, multi-operation patients, and more — is in Section 15.

---

## 3. Repository Structure

```
inspire-analysis-thrisha/
└── src/
    ├── dnn_mortality_pipeline.py     ← MAIN FILE — run this
    ├── dnn_mortality_data.py         ← data loader for real JSON subjects
    ├── charts.py                     ← AUROC and AUPRC plot functions
    ├── gbm_mortality.py              ← Saranya's GBM baseline
    ├── nela.py                       ← NELA clinical scorecard
    ├── frailty_hfrs.py               ← Hospital Frailty Risk Score (HFRS)
    ├── inspire_analysis_department.py ← exploratory plots by surgical dept
    ├── subject.py                    ← Subject class for folder-based data
    ├── inspire_dataset.py            ← reads the full inspire_subjects/ folder
    └── stopwatch.py                  ← timing utility
```

**Data (not in repo — stored separately):**
```
inspire_subjects_small/          ← 30-patient subset used for development
    survived/                    ← 20 JSON files (label = 0)
    died/                        ← 10 JSON files (label = 1)
```

---

## 4. Dataset Structure — INSPIRE

### Scale

| Table | Size | Description |
|---|---|---|
| Operations | 130,960 rows | One row per surgery |
| Labs | 1,048,575 rows | Blood test results over time |
| Ward vitals | Large | Bedside measurements on the ward |
| Intraop vitals | Large | Minute-by-minute measurements during surgery |
| Medications | 1,048,575 rows | Drug administrations |
| Diagnoses | 1,048,575 rows | ICD-10 codes |

### Full dataset mortality
- **Total patients:** ~99,886
- **Deaths:** 942 (see note in Section 1 re: a conflicting figure of 938 seen elsewhere)
- **Death rate:** 0.95%
- **pos_weight for loss function:** ~105 (98,944 survived / 942 died)

### Subset used for development
- **Total:** 30 patients (after filtering)
- **Survived:** 20 (in `survived/` folder)
- **Died:** 10 (in `died/` folder)
- **Actually loaded:** 29 (1 skipped — too sparse)

### Time reference
All `chart_time` values are in **minutes** relative to each patient's hospital admission (time = 0). Negative values mean before admission (outpatient measurements). Positive values mean after admission.

```
chart_time = -23395  → 16.2 days before admission (outpatient test)
chart_time = 0       → moment of hospital admission
chart_time = 710     → 710 minutes after admission = OR entry (orin_time)
chart_time = 825     → patient leaves OR (orout_time)
chart_time = 22535   → patient dies (15.1 days after admission)
```

### 30-day mortality label definition

```python
label = 1 if inhosp_death_time < orout_time + (30 × 24 × 60) else 0
# i.e. died within 30 days of leaving the operating room
```

The label is already encoded in the folder structure (`survived/` = 0, `died/` = 1).

### Prediction window — current implementation, scope under review

The current implementation only uses data from **5 days before surgery**:
```
window_start = orin_time - (5 × 24 × 60)   = orin_time - 7200 minutes
window_end   = orin_time - 1
```

> ⚠️ **Open scope decision:** this project is described as *peri-operative*, but the current pipeline is *pre-operative only*. Whether the model should also ingest intra-operative vitals (the 72-parameter `vitals` table, Section 11) — making it a true peri-operative model usable for post-operative risk stratification, rather than a pre-op decision-support tool — has **not yet been decided**. Do not assume either direction in downstream work (paper drafts, feature audits) until this is resolved. See Section 15, item 5.

---

## 5. Subject JSON File — Complete Format

Each patient is stored as one JSON file named `<subject_id>.json`. The file has **7 top-level keys:**

```json
{
  "subject_id": "100033460",

  "operations": [
    {
      "op_id":           "475179926",
      "subject_id":      "100033460",
      "hadm_id":         "240762127",
      "case_id":         "15178",
      "opdate":          "0",
      "age":             "80",
      "sex":             "F",
      "weight":          "0",
      "height":          "150",
      "race":            "Asian",
      "asa":             "3",
      "emop":            "1",
      "department":      "GS",
      "antype":          "General",
      "icd10_pcs":       "0DJ00",
      "orin_time":       "710",
      "orout_time":      "825",
      "opstart_time":    "735",
      "opend_time":      "815",
      "admission_time":  "0",
      "discharge_time":  "23035",
      "anstart_time":    "720",
      "anend_time":      "820",
      "cpbon_time":      "",
      "cpboff_time":     "",
      "icuin_time":      "825",
      "icuout_time":     "22710",
      "inhosp_death_time":   "22535",
      "allcause_death_time": "41760"
    }
  ],

  "labs": [
    {"subject_id": "100033460", "chart_time": "-23395", "item_name": "ptinr",    "value": "0.9"},
    {"subject_id": "100033460", "chart_time": "-23395", "item_name": "aptt",     "value": "29.1"},
    {"subject_id": "100033460", "chart_time": "-23380", "item_name": "sodium",   "value": "135.0"},
    ...
  ],

  "ward_vitals": [
    {"subject_id": "100033460", "chart_time": "-23490", "item_name": "hr",       "value": "96.0"},
    {"subject_id": "100033460", "chart_time": "-23300", "item_name": "hr",       "value": "80.0"},
    ...
  ],

  "vitals": [
    {"op_id": "475179926", "subject_id": "100033460", "chart_time": "700", "item_name": "bt",      "value": "22.0"},
    {"op_id": "475179926", "subject_id": "100033460", "chart_time": "700", "item_name": "art_mbp", "value": "6.0"},
    ...
  ],

  "diagnoses": [
    {"subject_id": "100033460", "chart_time": "-231840", "icd10_cm": "N18"},
    {"subject_id": "100033460", "chart_time": "-24480",  "icd10_cm": "N17"},
    {"subject_id": "100033460", "chart_time": "0",       "icd10_cm": "K65"},
    {"subject_id": "100033460", "chart_time": "20160",   "icd10_cm": "C18"},
    ...
  ],

  "medications": [
    {"subject_id": "100033460", "chart_time": "21720", "drug_name": "thiamine",
     "route": "iv", "drug_name2": "", "drug_name3": "",
     "atc_code": "A11DA01", "atc_code2": "", "atc_code3": ""},
    ...
  ]
}
```

### Key field reference

| Field | Table | Meaning |
|---|---|---|
| `chart_time` | all | Minutes from admission. Negative = before admission |
| `item_name` | labs, ward_vitals, vitals | What was measured (e.g. `glucose`, `hr`) |
| `value` | labs, ward_vitals, vitals | Measured value as string — must cast to float |
| `orin_time` | operations | Patient enters operating room |
| `orout_time` | operations | Patient leaves operating room |
| `inhosp_death_time` | operations | Time of death (empty string if survived) |
| `emop` | operations | Emergency operation: `"1"` = yes, `"0"` = no |
| `asa` | operations | ASA physical status class (1–5) |
| `icd10_cm` | diagnoses | ICD-10 diagnosis code (used for frailty score) |
| `icd10_pcs` | operations | ICD-10 procedure/operation code |
| `atc_code` | medications | WHO ATC drug classification code |

---

## 6. Real Patient Example — Subject 100033460

This is a real INSPIRE patient in the `died/` folder. This example is used throughout the codebase for testing.

### Patient summary

| Field | Value | Clinical meaning |
|---|---|---|
| Subject ID | 100033460 | Unique patient identifier |
| Age | 80 | Elderly |
| Sex | Female | F |
| Department | GS | General Surgery |
| Emergency | Yes (emop=1) | Brought in urgently — no pre-selection |
| ASA class | 3 | Severe systemic disease |
| Operation | 0DJ00 | Bowel inspection (colonoscopy/inspection) |
| Anaesthesia | General | Full general anaesthesia |
| OR entry | minute 710 | ~11.8 hours after admission |
| OR exit | minute 825 | 75-minute operation |
| ICU admission | minute 825 | Went straight to ICU post-op |
| ICU discharge | minute 22,710 | ~15.8 days in ICU |
| Death | minute 22,535 | ~15.1 days after admission |
| 30-day label | **1 (DIED)** | 22535 < 825 + 43200 ✓ |

### Why she died (diagnoses)

| ICD-10 | Meaning | Times recorded |
|---|---|---|
| N18 | Chronic kidney disease | 6 times (pre-existing) |
| N17 | Acute kidney failure | 1 time (on admission) |
| K65 | Peritonitis | On admission — reason for emergency surgery |
| C18 | Colon cancer | Post-operatively |
| S00 | Superficial injury of head | Incidental |

An 80-year-old woman with chronic kidney disease (N18) and colon cancer (C18) presented as an emergency with peritonitis (K65 — infection in the abdominal cavity). Her kidneys went into acute failure (N17). She spent 15 days in ICU on dialysis (CRRT active in ward vitals) and died.

*This patient is a useful worked example for the ICD-10/mortality exploration in Section 15 — she illustrates the multi-code-per-patient case (5 distinct codes) and a case where diagnosis, emergency status, and outcome all align clinically.*

### Her measurements

| Source | Total rows | Types available |
|---|---|---|
| Labs | 1,861 | 35 unique lab types |
| Ward vitals | 7,567 | 10 unique types |
| Intraop vitals | 457 | 27 unique types recorded for this patient (72 total exist in the schema — see Section 11) |
| Diagnoses | 12 | 5 unique ICD-10 codes |
| Medications | 399 | 8 ATC first-level drug classes |

### In the 5-day pre-op window (minute -6490 to 710)

| Source | Rows in window | Types in window |
|---|---|---|
| Labs | 80 | 33 types |
| Ward vitals | 814 | 7 types |

### Lab values at key moments (showing clinical deterioration)

```
Creatinine:
  chart_time  535:  5.55 mg/dL   (pre-op, day of surgery — severely elevated, normal < 1.2)
  → Chronic + acute kidney failure confirmed

Glucose:
  chart_time  510:  78.0 mg/dL
  chart_time  525: 264.0 mg/dL   (spiked to 264 — stress hyperglycaemia)
  chart_time  535:  84.0 mg/dL

Sodium:
  chart_time  510: 128.0 mEq/L   (low — normal 136-145, hyponatraemia)
  chart_time  535: 132.0 mEq/L

Potassium:
  chart_time  510:   5.3 mEq/L   (elevated — normal 3.5-5.0, kidney failure)
```

---

## 7. Dataset Subset Structure on Disk

```
inspire_subjects_small/
├── survived/                        ← 20 patients, label = 0
│   ├── 100002413.json
│   ├── 100016333.json
│   ├── 100001820.json
│   ├── 100024770.json
│   ├── 100026753.json
│   └── ... (15 more)
└── died/                            ← 10 patients, label = 1
    ├── 100403813.json
    ├── 100528073.json
    ├── 100516694.json
    ├── 100316372.json
    ├── 100407302.json
    └── ... (5 more)
```

**Zip file path in Colab:** After uploading `inspire_subjects_small.zip` and extracting:
```
/content/inspire_subjects_small/inspire_subjects_small/survived/
/content/inspire_subjects_small/inspire_subjects_small/died/
```

Note the double nesting — the zip contains a subfolder with the same name.

---

## 8. Current Pipeline Architecture

### Data flow — end to end

```
inspire_subjects_small/survived/*.json  (label=0)
inspire_subjects_small/died/*.json      (label=1)
                ↓
load_real_subjects(json_dir, feature_columns, days_before_operation=5)
  - reads survived/ and died/ folders, label from folder name
  - extracts 5-day pre-op window: [orin_time-7200, orin_time-1]
  - pulls matching item_name records from labs + ward_vitals
  - returns dict: {subject_id: {timeseries: {...}, label: 0/1}}
  - currently: 29 loaded (10 died, 19 survived), 1 skipped too-sparse
                ↓
align_time_series(time_series, full_length=None)
  - builds one row per minute from min_observed to max_observed
  - fills gaps with linear interpolation
  - adds mask column per feature: 1.0=observed, 0.0=interpolated
  - output: DataFrame, shape (num_minutes, num_features*2)
                ↓
normalize_data(df, feature_columns, scaler)
  - StandardScaler: subtract mean, divide by std
  - scaler FITTED ON TRAIN DATA ONLY (prevents data leakage)
  - test data TRANSFORMED using train scaler
                ↓
          ┌─────────────────────────────────┐
          │         PHASE 1                 │
          │   Unsupervised pretraining      │
          │   No mortality labels used      │
          └─────────────────────────────────┘
                ↓
create_sequences(df, seq_length=180, feature_columns, mask_columns)
  - slides a 180-minute window along the aligned series
  - 1440-minute series → 1,261 windows of shape (180, 14)
  - all training subjects combined → ~25,000 windows total
                ↓
TimeSeriesTransformer (mode='autoencode')
  ┌─────────────────────────────────────────────────┐
  │  input_projection:  Linear(14 → 14)             │
  │  + positional encoding (sinusoidal, shape 180×14)│
  │  transformer_encoder:                            │
  │    5 layers                                     │
  │    7 attention heads  (14 ÷ 7 = 2 per head)     │
  │    dim_feedforward = 128                        │
  │    dropout = 0.1                                │
  │  output_projection: Linear(14 → 7)              │
  │    reconstructs 7 feature values (not masks)    │
  │  Loss: MSE(reconstruction, actual_features)     │
  └─────────────────────────────────────────────────┘
  Result: loss 0.238 → 0.005 over 10 epochs ✓
                ↓
          ┌─────────────────────────────────┐
          │         PHASE 2                 │
          │   Supervised classification     │
          │   Encoder UNFROZEN              │
          │   Whole network fine-tuned      │
          └─────────────────────────────────┘
                ↓
SubjectDataset
  - full 1440-minute aligned+normalised series per patient
  - padded to global_length=1440 with zeros
  - attention mask: 1=real, 0=padded
                ↓
TimeSeriesTransformer (mode='classify')
  ┌─────────────────────────────────────────────────┐
  │  Same encoder as Phase 1 (unfrozen)             │
  │  Input: (batch, 1440, 14)                       │
  │  Encoder output: (batch, 1440, 14)              │
  │  Mean pool over 1440 time steps                 │
  │    → (batch, 14)  ← THE EMBEDDING               │
  │  classifier: Linear(14 → 1)                     │
  │    → sigmoid → mortality probability            │
  │  Loss: BCEWithLogitsLoss                        │
  │    pos_weight = 1.86 (19 survived / 10 died)    │
  │    Learning rate = 0.0001 (lower to protect      │
  │    pretrained encoder weights)                  │
  └─────────────────────────────────────────────────┘
  Result: loss 0.675 → 0.563 over 20 epochs, still dropping ✓
                ↓
evaluate_model()
  - AUROC = 0.7778 on 9 test patients (3 died, 6 survived)
  - Best F1 = 0.667 at threshold 0.17
  - Saves: embeddings.png, auroc.png, auprc.png
```

### Model parameter count

| Component | Parameters |
|---|---|
| input_projection (14×14 + 14 bias) | 210 |
| transformer_encoder (5 layers × ~4,000) | ~20,000 |
| output_projection (14×7 + 7 bias) | 105 |
| classifier (14×1 + 1 bias) | 15 |
| **Total** | **23,440** |

---

## 9. Current Config Values

All in `main()` inside `dnn_mortality_pipeline.py`:

```python
USE_REAL_DATA          = True
JSON_DIR               = "/content/inspire_subjects_small/inspire_subjects_small"
FEATURE_COLUMNS        = ['glucose', 'potassium', 'sodium', 'creatinine',
                           'hr', 'spo2', 'nibp_sbp']
DAYS_BEFORE_OPERATION  = 5
UNFREEZE_ENCODER       = True
AUTOENCODER_EPOCHS     = 10
CLASSIFIER_EPOCHS      = 20

# In train_autoencoder():
nhead                  = 7          # must divide evenly into num_features
num_layers             = 5
dim_feedforward        = 128
dropout                = 0.1
learning_rate          = 0.001      # autoencoder

# In train_classifier():
learning_rate          = 0.0001     # lower to protect pretrained weights

# In main():
global_length          = min(max_length, 1440)   # cap at 24 hours for GPU memory
```

### The nhead rule

`nhead` must divide evenly into `num_features` where `num_features = len(FEATURE_COLUMNS) × 2`:

| FEATURE_COLUMNS count | num_features | Valid nhead values |
|---|---|---|
| 4 features | 8 | 1, 2, 4, 8 |
| 7 features (current) | 14 | 1, 2, 7, 14 |
| 10 features | 20 | 1, 2, 4, 5, 10, 20 |
| 16 features | 32 | 1, 2, 4, 8, 16, 32 |
| 30 features | 60 | 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60 |
| 33 features | 66 | 1, 2, 3, 6, 11, 22, 33, 66 |

---

## 10. Current Results

| Run | Patients | Features | Epochs | AUROC | Notes |
|---|---|---|---|---|---|
| Synthetic frozen encoder | 300 fake | 4 | 10 | ~0.50 | Encoder frozen — classifier learned nothing |
| Synthetic unfrozen | 300 fake | 4 | 3 | ~0.60 | Unfreezing fixed the problem |
| Real data first run | 20 real | 4 | 3 | 0.8889 | Accidentally perfectly balanced 10/10 |
| Real data | 22 real | 4 | 3 | 0.6667 | More realistic split |
| Real data all 30 | 30 real | 7 | 10 | 0.8571 | Best run — 10 died / 20 survived |
| Real data latest | 29 real | 7 | 20 | 0.7778 | Classifier still not converged |

**Why AUROC fluctuates:** Test set is only 6–10 patients. One wrong prediction moves AUROC by ~0.11. These numbers will stabilise only with the full 99,886-patient dataset. The model is genuinely learning — classifier loss drops every epoch.

---

## 11. All Available Features — What Is Used vs What Exists

### Labs (from labs list in JSON) — 38 types in the full schema

| Feature | System | Currently used | Measurement count (example patient, Section 6) |
|---|---|---|---|
| glucose | Metabolic | ✅ | 144 |
| potassium | Renal | ✅ | 111 |
| sodium | Renal | ✅ | 111 |
| creatinine | Renal | ✅ | 42 |
| hb | Haematology | ❌ | 110 |
| hct | Haematology | ❌ | 116 |
| wbc | Haematology | ❌ | 44 |
| albumin | Metabolic | ❌ | 29 |
| ast | Metabolic | ❌ | 28 |
| alt | Metabolic | ❌ | 28 |
| total_bilirubin | Metabolic | ❌ | 28 |
| alp | Metabolic | ❌ | 28 |
| total_protein | Metabolic | ❌ | 28 |
| lacate | Metabolic | ❌ | 80 |
| crp | Haematology | ❌ | 21 |
| ph | Respiratory | ❌ | 92 |
| pao2 | Respiratory | ❌ | 91 |
| paco2 | Respiratory | ❌ | 91 |
| hco3 | Respiratory | ❌ | 91 |
| sao2 | Respiratory | ❌ | 91 |
| be | Respiratory | ❌ | 82 |
| bun | Renal | ❌ | 35 |
| calcium | Renal | ❌ | 36 |
| chloride | Renal | ❌ | 39 |
| phosphorus | Renal | ❌ | 36 |
| ica | Renal | ❌ | 17 |
| platelet | Haematology | ❌ | 38 |
| ptinr | Haematology | ❌ | 17 |
| aptt | Haematology | ❌ | 16 |
| fibrinogen | Haematology | ❌ | 13 |
| lymphocyte | Haematology | ❌ | 38 |
| seg | Haematology | ❌ | 37 |
| troponin_i | Cardiovascular | ❌ | 5 |
| troponin_t | Cardiovascular | ❌ | — |
| ck | Cardiovascular | ❌ | 4 |
| ckmb | Cardiovascular | ❌ | 3 |
| d_dimer | Haematology | ❌ | — |
| hba1c | Metabolic | ❌ | — |

*(38 total; two — `troponin_t` and `hba1c` — weren't observed for the Section 6 example patient, hence no count.)*

### Ward vitals (from ward_vitals list in JSON) — 16 types

| Feature | System | Currently used | Measurement count (example patient) |
|---|---|---|---|
| hr | Cardiovascular | ✅ | 908 |
| spo2 | Respiratory | ✅ | 882 |
| nibp_sbp | Cardiovascular | ✅ | 941 |
| nibp_dbp | Cardiovascular | ❌ | 940 |
| nibp_mbp | Cardiovascular | ❌ | 939 |
| rr | Respiratory | ❌ | 1,084 |
| fio2 | Respiratory | ❌ | 639 |
| bt | Metabolic | ❌ | 441 |
| crrt | Renal | ❌ | 792 (dialysis active!) |
| uo | Renal | ❌ | 1 |
| gcs_e | Neurological | ❌ | — |
| gcs_m | Neurological | ❌ | — |
| gcs_v | Neurological | ❌ | — |
| iabp | Cardiovascular | ❌ | — |
| ecmo | Respiratory | ❌ | — |
| vent | Respiratory | ❌ | — |

*(Ward vitals include the 3 Glasgow Coma Scale components and device-use flags (`crrt`, `iabp`, `ecmo`, `vent`) not previously catalogued in this table — see Section 15 item 4 re: the neurological system grouping these motivate.)*

### Intraoperative vitals (from vitals list in JSON) — 72 types

*Not currently used in the pipeline.* This table requires the intra-operative window (`orin_time` to `orout_time`) rather than the pre-op window currently implemented — see the open scope decision flagged in Section 4. Whether these 72 parameters are added depends on that decision, not on any technical limitation.

```
aft, air, alb20, alb5, art_dbp, art_mbp, art_sbp, bis, bt, cbro2, ci, cryo, cvp,
d10w, d50w, d5w, dobui, dopai, ebl, eph, epi, epii, etco2, etdes, etgas, etiso,
etsevo, ffp, fio2, ftn, hes, hns, hr, hs, mdz, minvol, mlni, n2o, nepi, nibp_dbp,
nibp_mbp, nibp_sbp, ns, ntgi, o2, pap_dbp, pap_mbp, pap_sbp, pc, peep, pepi, phe,
pheresis, pip, pmean, ppf, ppfi, pplat, psa, rbc, rfti, rr, sft, spo2, sti, stii,
stiii, stv5, svi, uo, vaso, vt
```

*(Corrected from an earlier version of this document that listed only 27 intra-op types — the full schema has 72.)*

### To expand FEATURE_COLUMNS to all useful pre-operative labs + ward vitals:

```python
FEATURE_COLUMNS = [
    # Renal (5)
    'creatinine', 'bun', 'sodium', 'potassium', 'chloride',
    # Cardiovascular ward (4)
    'hr', 'nibp_sbp', 'nibp_dbp', 'nibp_mbp',
    # Respiratory labs (6)
    'pao2', 'paco2', 'ph', 'hco3', 'be', 'sao2',
    # Respiratory ward (3)
    'spo2', 'rr', 'fio2',
    # Metabolic/hepatic labs (6)
    'glucose', 'albumin', 'ast', 'alt', 'total_bilirubin', 'lacate',
    # Haematology labs (5)
    'hb', 'hct', 'wbc', 'platelet', 'crp',
    # Temperature (1)
    'bt',
]
# Total: 30 features → num_features = 60 → use nhead=6 (60÷6=10) or nhead=4 (60÷4=15)
```

---

## 12. How to Run — Colab Workflow

### Every new session (cells run in order):

```python
# Cell 1 — GPU check
import torch
print(torch.cuda.is_available())       # must print True
print(torch.cuda.get_device_name(0))   # must print Tesla T4

# Cell 2 — Clone repo
!git clone https://github.com/thrisharajkumar/inspire-analysis-thrisha.git
import os
os.chdir('/content/inspire-analysis-thrisha/src')

# Cell 3 — Upload data zip
from google.colab import files
uploaded = files.upload()  # select inspire_subjects_small.zip

# Cell 4 — Extract data
import zipfile, os
zip_path = '/content/inspire-analysis-thrisha/src/inspire_subjects_small.zip'
extract_dir = '/content/inspire_subjects_small'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extract_dir)
contents = os.listdir(extract_dir)
if len(contents) == 1 and os.path.isdir(os.path.join(extract_dir, contents[0])):
    extract_dir = os.path.join(extract_dir, contents[0])
print('survived:', len(os.listdir(os.path.join(extract_dir, 'survived'))))
print('died:    ', len(os.listdir(os.path.join(extract_dir, 'died'))))
print('DATA PATH:', extract_dir)
# Should print: survived: 20, died: 10
# DATA PATH: /content/inspire_subjects_small/inspire_subjects_small

# Cell 5 — Pull latest code and run
!git pull
!python dnn_mortality_pipeline.py

# Cell 6 — View plots
from IPython.display import Image, display
for f in ['embeddings.png', 'auroc.png', 'auprc.png']:
    path = f'/content/inspire-analysis-thrisha/src/{f}'
    print(f'--- {f} ---')
    display(Image(path))
```

### After making changes in VS Code and pushing:

```python
# Just run these two:
!git pull
!python dnn_mortality_pipeline.py
```

---

## 13. Known Issues and Fixes Applied

| Issue | Symptom | Fix applied |
|---|---|---|
| Scapy import | `ModuleNotFoundError` on startup | Deleted line `from scapy.layers.tls.crypto.groups import modp2048` |
| Frozen encoder | Classifier output ~0.47 for every patient, AUROC ~0.5 | Set `UNFREEZE_ENCODER = True`, changed to `param.requires_grad = True` |
| nhead not dividing num_features | `AssertionError: embed_dim must be divisible by num_heads` | Changed nhead=8 to nhead=7 (for 14-dim input) |
| GPU out of memory | `CUDA out of memory: Tried to allocate 9.11 GiB` | Added `global_length = min(max_length, 1440)` cap |
| Wrong not_survived_pct | Negative ratio crashed train/test split | Changed `num_died/num_survived` to `num_died/total` |
| Tensor slow warning | `Creating tensor from list of ndarrays is slow` | Changed `torch.tensor(sequences)` to `np.array(sequences)` then `torch.from_numpy()` |
| JSON data in survived/died folders | `FileNotFoundError: No .json files found` | Updated `load_real_subjects()` to detect and read subfolder layout |
| ZIP nested folder | `extract_dir` pointed at wrong level | Added single-subfolder detection logic in extraction cell |
| Too-sparse subjects skipped | 8–10 patients lost | Reduced `min_observations=2` to `min_observations=1` |
| MPS (Apple Silicon) NaN | NaN values during classifier training on Mac | `get_device()` returns `"cuda"` when available, bypasses MPS |

---

## 14. Research Direction and Next Steps

### Immediate (to get meaningful results)

**1. Get full dataset from James**

Current subset has 30 patients — not enough for stable AUROC. Full dataset has 99,886 patients, 942 deaths. Change in pipeline:
```python
JSON_DIR = "/path/to/full/inspire_subjects"
global_length = min(max_length, 7200)   # allow full 5-day window
```
The `pos_weight` is computed automatically — no other changes needed.

**2. Expand to all features**

Change `FEATURE_COLUMNS` to include all 30 useful features (see Section 11), update `nhead=6`, run.

**3. More classifier epochs**

Loss was still dropping at epoch 20. Run with `CLASSIFIER_EPOCHS = 50` and watch when it plateaus.

### Core research contribution — system-separated embedding architecture

Your professor's vision: instead of one transformer for everything, train **separate encoders per physiological system**. Each system produces its own embedding. A surgeon can then see which system drove the prediction.

```
Current (black box):
  7 features → 1 transformer → 1 embedding → mortality

Target (explainable):
  Renal features        → Encoder 1 → renal_embedding
  Cardiovascular features → Encoder 2 → cardio_embedding
  Respiratory features  → Encoder 3 → resp_embedding
  Metabolic features    → Encoder 4 → metabolic_embedding
                                 ↓
                    concatenate embeddings
                                 ↓
                    NAM classifier (one shape function per system)
                                 ↓
        "Renal: CRITICAL | Respiratory: CONCERN | Cardio: OK | Metabolic: OK"
        → mortality risk = 0.84
```

**Clinical feature groups (4-system core; see Section 15 item 4 for the fuller 6-system version including haematology and neurological):**

| System | Features |
|---|---|
| Renal | creatinine, bun, sodium, potassium, chloride, uo, crrt |
| Cardiovascular | hr, nibp_sbp, nibp_dbp, nibp_mbp, troponin_i |
| Respiratory | spo2, rr, fio2, pao2, paco2, ph, hco3, be, sao2 |
| Metabolic/hepatic | glucose, albumin, ast, alt, total_bilirubin, lacate, bt |

### Explainability additions

**Concept Bottleneck Models (CBM)** — model predicts clinically validated scores before predicting mortality:
- AKI stage (0–3) from creatinine using KDIGO criteria
- Haemodynamic instability from BP + vasopressors using SOFA cardiovascular score
- Respiratory failure from PaO2/FiO2 ratio using Berlin criteria

**MC Dropout uncertainty** — 3-line change, keep dropout on at test time, run 50 forward passes, output confidence interval instead of single probability.

**TimeSHAP** — extends SHAP to time series, shows which features at which time steps drove each prediction.

**Conformal prediction** — mathematically guaranteed prediction intervals:
```
Current: "mortality = 0.73"
Target:  "mortality ∈ [0.58, 0.84] with 90% guaranteed coverage"
```

### Comparison models to run

- **NELA** (`nela.py`) — clinical baseline, no training needed
- **GBM** (`gbm_mortality.py`) — Saranya's XGBoost, 18 pre-op features
- **DNN transformer** — this project, expanding to full feature set
- **GRU** — add as comparison, faster than transformer, fewer parameters
- **TCN (Temporal Convolutional Network)** — mentioned in professor's meeting notes, fully parallelisable on GPU

---

## 15. Research Exploration Roadmap

This section captures every open thread being explored for this project, organised by theme. It is not sequential — items are being investigated in parallel — but "Current focus" is the near-term priority everything else feeds into (see also Section 2, Aims and Objectives).

### Current focus

**1. Model interpretability**
The core deliverable right now is a model that is not just accurate but explainable — tied specifically to the two clinical questions below rather than interpretability in the abstract.

**2. ICD-10 codes and mortality rate**
- Explore diagnosis ICD-10 codes and operation-type ICD-10 codes separately — they answer different questions (what's wrong with the patient vs. what's being done to them).
- A patient may have more than one ICD-10 code (see the Section 6 example patient — 5 distinct codes) — need to decide how multi-code patients are represented (one-hot per code? code count? weighted by severity? grouped by ICD-10 chapter?).
- Operation type vs. mortality — does procedure complexity/category predict risk independently of diagnosis?
- This connects to the existing HFRS frailty work (`docs/INSPIRE_Project_Notes.md` §11), which already uses 109 ICD-10 codes for frailty scoring — worth checking overlap/redundancy between "frailty codes" and "high-mortality-risk codes" once this analysis exists.

**3. Department and mortality rate risk**
- 15 surgical departments in INSPIRE — mortality rate likely varies substantially by department (case mix, patient acuity, procedure type all differ).
- Needs to be analysed jointly with ICD-10 (a department's mortality rate is partly explained by which operations/diagnoses it sees) rather than as an independent factor.
- Should feed into visualisations: mortality rate by department, stratified by scheduled vs. emergency, and by ICD-10 category.

### Feature categorisation

**4. Organ-system grouping of all 126 parameters**
Full categorisation of `labs` (38), `ward_vitals` (16), and intra-operative `vitals` (72) into physiological systems. Extending the 4-system core (Section 14) with the ward-vitals fields that don't fit cleanly there:

| System | Labs | Ward vitals |
|---|---|---|
| Renal | bun, calcium, chloride, creatinine, ica, phosphorus, potassium, sodium | crrt, uo |
| Cardiovascular | ck, ckmb, troponin_i, troponin_t | hr, nibp_sbp, nibp_dbp, nibp_mbp, iabp |
| Respiratory | be, hco3, paco2, pao2, ph, sao2 | fio2, rr, spo2, vent, ecmo |
| Metabolic/hepatic | albumin, alp, alt, ast, glucose, hba1c, lacate, total_bilirubin, total_protein | bt |
| Haematology/coagulation | aptt, crp, d_dimer, fibrinogen, hb, hct, lymphocyte, platelet, ptinr, seg, wbc | — |
| Neurological | — | gcs_e, gcs_m, gcs_v |

This groups both the *input features* and, downstream, the *model's per-system predictions* — i.e. the same categorisation drives both feature engineering and the interpretability output (Section 14).

**5. Using all 126 parameters in the model**
Currently 7 of 126 are used. Full audit needed (coverage, missingness, signal) before finalising the feature set. This is gated on the open scope decision (Section 4): whether the model stays pre-operative only (54 candidate parameters: 38 labs + 16 ward_vitals), or becomes a full peri-operative model also using the 72 intra-operative `vitals` parameters. **Decision not yet made — do not assume either direction in downstream work until this is resolved.**

### Class imbalance and data quality

**6. Sampling methods for 99,886 vs. 942 (0.95% mortality)**
Candidates to evaluate against each other:
- Oversampling / undersampling (basic)
- SMOTE and variants
- Generative approaches: GANs, VAEs, diffusion models, Gaussian-based synthetic sampling
- Tabular-data-specific generative methods
- Cost-sensitive loss functions (already partially in use via `pos_weight` — worth comparing against the above rather than assuming it's sufficient at full scale)
- Simply acquiring more labelled death cases, if feasible

**7. Missing data handling**
- Linear interpolation (already in use — Section 8)
- Fitting a model to predict/impute missing features, rather than just interpolating
- Laplace smoothing where relevant (e.g. sparse categorical counts)
- Stratifying the data and handling near-zero-probability cases carefully — avoid the model treating a rare event as effectively impossible

**8. Data distribution analysis**
General EDA — distributions, outliers, correlations — across the full parameter set, extending the correlation work already done (`docs/INSPIRE_Project_Notes.md` §11: frailty–sodium, frailty–albumin correlations).

**9. HFRS — what it is and a fix needed**
HFRS = Hospital Frailty Risk Score (Gilbert et al. 2018, Lancet). It's 109 ICD-10 codes, each with a point value, derived from a cluster analysis of frail vs. non-frail hospital patients. Sum the points from a patient's diagnosis codes to get a score: **<5 = low, 5–15 = intermediate, >15 = high** (already correct in `subject.py:compute_hfrs()`).

The published method only counts diagnoses from the **prior 2 years**, for patients aged 75+. Our current code uses **all** of a patient's diagnoses ever, no time limit, no age cutoff — this needs fixing before HFRS results can be trusted.

Plan: use HFRS two ways — (1) as its own baseline, no training, like NELA, and (2) as an extra feature added to GBM/DNN. Evaluate with AUROC + AUPRC, compared with and without HFRS, and split by scheduled vs. emergency surgery (ties into item 10 below).

### Clinical structure

**10. Conditional dependencies**
E.g. frailty is largely uninformative once you know surgery was scheduled (a surgeon has already selected for fitness), but highly informative for emergency surgery where no such selection occurred. This hypothesis (framed in `docs/INSPIRE_Project_Notes.md` §11) needs a scheduled-vs-emergency stratified analysis to test properly, and may generalise to other conditional relationships beyond frailty.

**11. Multi-operation patients**
Does a patient with two or more recorded operations carry different mortality risk than a single-operation patient? Needs its own cohort split and analysis — this is a patient-level question, separate from the per-operation prediction the current pipeline makes.

### Modelling

**12. DNN generalisation and robustness**
- Shared representation learning — how much should the organ-system encoders share vs. specialise?
- Robustness — does the model degrade gracefully with missing/noisy inputs, or brittle?
- Semi-supervised learning — the current autoencoder pretraining phase (Section 8) is already a step in this direction; worth exploring further given how few labelled deaths exist relative to the full cohort.

### Suggested order of attack

Given the current focus is interpretability tied to ICD-10 and department, a reasonable sequence is:
1. EDA — ICD-10 code distributions, department distributions, both vs. mortality (items 2, 3, 8)
2. Organ-system feature categorisation for all 126 parameters (items 4, 5) — pending the scope decision in Section 4
3. Missing data / distribution handling for whichever features EDA selects (items 6, 7)
4. HFRS fix and evaluation (item 9)
5. Conditional dependency and multi-operation analysis (items 10, 11)
6. Model architecture and generalisation work (item 12), building on the current two-phase pipeline (Section 8)

---

## 16. Key Papers and References

| Paper | Why it matters |
|---|---|
| Shickel et al. 2023, Scientific Reports, doi:10.1038/s41598-023-27418-5 | Closest published comparison — same problem, 56,242 patients, AUROC 0.92. Has MC Dropout and integrated gradients. Gaps: single shared representation, no system organisation, no frailty, binary outcomes only. |
| Koh et al. 2020 (ICML) — Concept Bottleneck Models | CBM architecture — model predicts clinical concepts before mortality |
| Choi et al. 2016 (NeurIPS) — RETAIN | Two-level attention for clinical time series (visit-level + feature-level) |
| Bento et al. 2021 (KDD) — TimeSHAP | SHAP adapted for time series — which feature at which time step mattered |
| Nature Medicine 2023 — Conformal Prediction in Clinical AI | Guaranteed uncertainty bounds for clinical deployment — exact citation not yet confirmed, needs verifying before use in the paper |
| Gilbert et al. 2018 — HFRS | Hospital Frailty Risk Score from ICD-10 codes — implemented in `frailty_hfrs.py` |
| INSPIRE dataset paper — Lee et al. | Must cite as data source — search "INSPIRE dataset perioperative Korean 2022"; exact citation not yet confirmed |

---

*README last updated: pipeline producing AUROC 0.78 on 29 real INSPIRE patients, classifier loss still converging at epoch 20, 7 features active. Pre-operative vs. full peri-operative model scope is an open decision (Section 4). Current research focus: ICD-10 and department vs. mortality interpretability (Sections 2, 15).*