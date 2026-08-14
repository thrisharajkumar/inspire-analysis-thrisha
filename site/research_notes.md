# Feature Audit & Data Exploration Findings

> This page documents what we've actually verified about the INSPIRE patient
> data, using real numbers from three self-contained audit scripts
> (`explore_features.py`, `audit_features.py`, `audit_static_categorical.py`)
> run against the real 30-patient subset. It replaces earlier single-patient
> guesses with numbers checked across all 30 patients.

---

## 1. How the patient JSON files are created

The raw INSPIRE dataset ships as six large CSV files (`operations.csv`,
`labs.csv`, `ward_vitals.csv`, `vitals.csv`, `diagnosis.csv`,
`medications.csv`), each with every patient's rows mixed together. The
conversion into one JSON file per patient (`inspire_dataset.py`, using the
`Subject` class in `subject.py`) works in four steps:

1. **Create one empty container per patient** — read `operations.csv`,
   collect every distinct `subject_id`, create one empty `Subject` object
   for each.
2. **Sort every row into the right container** — walk `labs.csv`,
   `ward_vitals.csv`, `vitals.csv`, `diagnosis.csv`, and `medications.csv`
   row by row; each row's `subject_id` determines which patient's
   container it's added to.
3. **Decide died vs. survived** — for each patient, check whether
   `inhosp_death_time` on their operation record is blank (survived) or has
   a value (died). No scoring involved — a direct field check.
4. **Write one JSON file per patient**, into `died/` or `survived/`
   depending on step 3.

This is why later steps (loading 30 patients, auditing features) are fast —
the slow, one-time cost of scanning tens of millions of CSV rows only ever
happens once, up front.

## 2. What's inside each patient JSON

Every file has **7 top-level keys**. Only two are currently used by the
model:

| Field | Shape | Used by `dnn_mortality_pipeline.py`? |
|---|---|---|
| `operations` | One row: age, sex, ASA class, emergency flag, department, timestamps | ❌ Not used |
| `labs` | Time series | ✅ Partially (7 of 38 types) |
| `ward_vitals` | Time series | ✅ Partially (3 of 16 types) |
| `vitals` (intraop) | Time series, surgery-only | ❌ Excluded by design (doesn't exist pre-op) |
| `diagnoses` | List of ICD-10 codes over time | ❌ Not used in this pipeline (used separately in `frailty_hfrs.py`) |
| `medications` | List of drug administrations over time | ❌ Not used |

## 3. The real feature schema (`parameters.csv`)

The dataset's own schema file lists **126 total feature types**:

| Table | Count | Usable for pre-op prediction? |
|---|---|---|
| `labs` | 38 | ✅ Yes |
| `ward_vitals` | 16 | ✅ Yes |
| `vitals` (intraop) | 72 | ❌ No — only exists during surgery |

**54 feature types (38 + 16) are usable for a pre-op prediction.** This
supersedes an earlier estimate of 45, which was based on only one patient's
recorded data rather than the full schema.

## 4. Per-patient completeness — the biggest real finding

Running `audit_features.py` across all 30 real patients (`parameters.csv` as
the schema) shows how many of the 54 usable features each patient actually
has data for, specifically inside the 5-day pre-op window:

- **Mean: 22.6 / 54**
- **Min: 1 / 54** (subject `100002413`)
- **Max: 47 / 54** (subjects `100316372`, `100403813`, `100573111`)

This is a very wide spread, and it isn't random — it tracks with **why the
patient was in hospital**. Low-completeness patients cluster in
lower-acuity departments (`OT` ophthalmology, `UR` urology, `OG`
obstetrics/gynaecology); high-completeness patients cluster in
higher-acuity departments (`GS` general surgery, `CTS`
cardio-thoracic surgery) where extensive pre-op bloodwork is standard.

### Coverage tiers (% of 30 patients with the feature in their pre-op window)

| Tier | Coverage | Features |
|---|---|---|
| Near-universal | 90–93% | `hr`, `bt`, `nibp_dbp`, `nibp_sbp`, `rr` (all ward vitals) |
| Solid core | 50–70% | `chloride, hb, hct, potassium, sodium, wbc, platelet, lymphocyte, seg, creatinine, bun, calcium, phosphorus, albumin, alp, alt, ast, glucose, total_bilirubin, total_protein, ptinr` |
| Patchy | 20–50% | `aptt, fibrinogen, crp, spo2, hco3, sao2, be, ck, ckmb, paco2, pao2, ph, nibp_mbp` |
| Basically absent | <20% | `ica, troponin_i, fio2, gcs_e, gcs_m, lacate, vent, hba1c, crrt, ecmo, gcs_v, iabp` |
| Never present pre-op | 0% | `d_dimer, troponin_t, uo` |

Two specific discoveries worth noting:

- `uo` (urine output) appears in 63% of patients *somewhere* in their file,
  but **0%** specifically pre-op — it seems to only get recorded once
  someone is already admitted/in ICU, not beforehand.
- `spo2` similarly drops from 93% "anywhere" to 40% "in the pre-op window,"
  for the same reason (monitors get attached on admission, not before).

### Department counts (too small to segment by yet)

```
GS: 6   OT: 5   UR: 5   CTS: 4   NS: 3
OG: 3   PS: 2   OL: 1   OS: 1
```

With 1–6 patients per department, department-level analysis isn't
statistically meaningful yet — worth revisiting once the full ~100k-patient
dataset is available.

## 5. A real fragility found in `align_time_series()`

When a feature has **zero** observations in a patient's window, the current
code fills the entire column with `0.0` and marks it as not-observed via the
mask column. Because normalization (`StandardScaler`) happens *after* this
fill step, that placeholder `0.0` becomes `(0 − mean) / std` — a fixed,
non-zero number, not a neutral value. For a patient like `100002413`
(1/54 features present), roughly 53 of their feature columns become this
artificial constant. The mask column does tell the model "this wasn't
real," so a well-trained model can in principle learn to discount it — but
for very sparse patients, their embedding risks mostly encoding *which
tests were absent* rather than real physiology. Worth being deliberate
about (e.g. impute with population median instead of 0) rather than
leaving as an unexamined side effect.

## 6. What's sitting unused: static facts, diagnoses, medications

Beyond the 54 time-series features, every patient record also has:

- **Static one-time facts** (`operations` table): age, sex, ASA class,
  emergency-operation flag, department. These have 100% coverage by
  definition (every patient has exactly one operation record) and include
  ASA class and age — both established, strong mortality predictors in
  real clinical scoring systems. Currently unused by the model entirely.
- **Diagnoses** (ICD-10 codes) — categorical, not time series. Already
  used separately for the Hospital Frailty Risk Score (`frailty_hfrs.py`)
  but not fed into the transformer.
- **Medications** (drug names + ATC codes) — also categorical, also
  unused by the transformer.

`audit_static_categorical.py` audits these three across all patients,
counting diagnoses/medications only if they occurred **before** `orin_time`
(to avoid leaking post-op information, e.g. a diagnosis made *because of*
the surgery, into a pre-op prediction).

## 7. Audit scripts reference

All three scripts are self-contained (standard library only, no repo
dependencies) so they can't break from unrelated code changes.

| Script | What it answers |
|---|---|
| `explore_features.py` | Simple pass: what feature types exist across patients, and how many records each has |
| `audit_features.py` | Per-patient completeness against the real `parameters.csv` schema, in-window vs. anywhere, missing-feature lists |
| `audit_static_categorical.py` | Operations (static facts), diagnoses, and medications — pre-op only |

Typical Colab usage:

```python
!python audit_features.py /content/inspire_subjects_small/inspire_subjects_small parameters.csv
!python audit_static_categorical.py /content/inspire_subjects_small/inspire_subjects_small
```

## 8. Open decisions (not yet made)

- **Which coverage tier to include in `FEATURE_COLUMNS`?** All 54 risks
  most patients being majority placeholder-filled noise (see Section 5).
  A defensible starting point is the near-universal + solid-core tiers
  (~26 features, all ≥50% coverage).
- **Should static facts (age, ASA, emop) be added** as extra per-patient
  inputs alongside the time-series features? Likely yes, given their
  established predictive value and 100% coverage.
- **Should diagnoses/medications use the same 5-day window** as labs, or
  a wider "anytime before admission" window (used currently in the audit
  script) since chronic conditions remain relevant risk factors regardless
  of when they were first diagnosed?
- **Department-level modeling** — parked until the full dataset is
  available (current per-department counts are too small: 1–6 patients).
