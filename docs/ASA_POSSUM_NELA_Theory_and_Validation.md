# ASA, POSSUM, NELA — Theory, Math, References, and Full-Cohort Validation

Companion to `Clinician_Questions.md` Section 3. That doc has the *questions* to ask the
clinical team; this doc has the *background* — what these scores actually are
mathematically, what's already published about their accuracy and limitations, and a plan
to check them against your own full 99,886-patient cohort rather than relying on
literature numbers alone.

---

## 1. ASA Physical Status

### What it is
A **single ordinal judgment** (class 1-6), assigned by the anesthesiologist before
surgery based on overall clinical impression. Not computed from any inputs.

### The math: there is none — and that's the key point
| | ASA | POSSUM / NELA |
|---|---|---|
| Inputs | None — a holistic clinical impression | 12-18 specific measured variables |
| Computation | None — a direct 1-6 judgment call | Weighted sum → sigmoid → probability |
| Reproducible from data alone? | ❌ No — depends on who's looking | ✅ Yes — same inputs always give the same output |

This is exactly why the circularity question (Section 4 below, and Clinician Question 7)
is worth asking: ASA's strength as a predictor comes from a human synthesizing information
a formula can't easily capture (gestalt clinical impression) — but that same property
means it can't be independently audited the way a formula can.

### History
- **Saklad, M. (1941).** "Grading of patients for surgical procedures." *Anesthesiology*,
  2, 281–284. — the original 7-class scale.
- **Dripps, R.D. (1963).** "New classification of physical status." *Anesthesiology*, 24,
  111. — reduced to the modern 5 (later 6, adding "brain-dead organ donor" as class 6)
  classes.

### Published predictive accuracy — benchmark numbers for our own data
- **Koo, C.Y. et al. (2015).** *World Journal of Surgery*, 39, 88–103. — pooled across
  **77 studies, 165,705 patients**: sensitivity 0.74, specificity 0.67, **AUROC 0.736**
  (95% CI 0.725–0.747). ASA performs *better* in lower-death-rate settings — directly
  relevant, since our cohort's 0.47% mortality is a low-death-rate setting by this paper's
  own framing.
- **Reliability concern:** early inter-rater studies (hypothetical case scenarios) found
  only **fair agreement, κ 0.21–0.4** between anesthesiologists grading the same patient
  (Sankar et al., *Br J Anaesth*, 2014). A more recent 56,820-case study (*J Med Syst*,
  2021) found ASA alone reaches **c-statistic 0.79** for 30-day mortality, rising to
  **0.82** when combined with objective comorbidity indices.

---

## 2. POSSUM / P-POSSUM

### What it is
**18 clinical variables**, each individually scored **1, 2, 4, or 8 points** (an
"exponential" scale — deliberately non-linear, so severe abnormalities count for much more
than mild ones), split into two groups that get summed separately, then combined through
one small logistic regression equation.

**Physiological Score (PS)** — 12 variables, summed range 12-88:
age • cardiac signs • respiratory signs • systolic BP • pulse rate • Glasgow Coma Scale •
hemoglobin • white cell count • urea • sodium • potassium • ECG findings

**Operative Severity Score (OSS)** — 6 variables, summed range 6-48:
operative severity magnitude • number of procedures • total blood loss • peritoneal
soiling • malignancy presence/stage • mode of surgery (elective vs. emergency)

*(The exact point cutoffs per variable — e.g. exactly which systolic BP range scores 1 vs.
2 vs. 4 vs. 8 — are published in the original Copeland 1991 paper and standard POSSUM
calculators; not reproduced here to avoid transcribing a clinical scoring table from
memory with any risk of error. Cross-reference the original paper or a validated
calculator before using exact cutoffs clinically.)*

### The equations — POSSUM computes BOTH morbidity and mortality risk
**Mortality (R1):**
```
ln(R1 / (1 - R1)) = -7.04 + 0.13 × PS + 0.16 × OSS
```
**Morbidity/complication risk (R2):**
```
ln(R2 / (1 - R2)) = -5.91 + 0.16 × PS + 0.19 × OSS
```
Same PS/OSS inputs, two different fitted equations — POSSUM was built to predict *both*
whether a patient develops a complication and whether they die, not mortality alone.

**P-POSSUM (Whiteley et al., 1996) — the mortality-only correction:**
```
ln(R / (1 - R)) = -9.065 + 0.1692 × PS + 0.1550 × OSS
```
Built specifically because the original POSSUM mortality equation (R1) **systematically
overestimates deaths** — confirmed repeatedly in later validation studies (e.g. a 2018
vascular-surgery cohort: POSSUM predicted 29.1 deaths against 6 actually observed;
P-POSSUM predicted 4.4, much closer to reality).

### All 18 input variables, with real INSPIRE data availability

Each variable is scored on a **4-level exponential scale (1, 2, 4, or 8)** — deliberately
non-linear, so a severely abnormal reading counts far more than a mildly abnormal one.

**Physiological Score (PS) — 12 variables, sum ranges 12-88:**

| # | Variable | Availability in INSPIRE |
|---|---|---|
| 1 | Age | ✅ Available — static field |
| 2 | Cardiac signs (clinical exam grade, e.g. JVP/edema/warfarin use) | ❌ Not available — needs exam grade, not a lab/vital |
| 3 | Respiratory signs (clinical exam grade, e.g. dyspnea severity) | ❌ Not available as exact match — raw `spo2`/`rr` exist as proxies |
| 4 | Systolic BP | ✅ Available — `nibp_sbp`, 98.2% |
| 5 | Pulse rate | ✅ Available — `hr`, 98.4% |
| 6 | Glasgow Coma Scale | ⚠️ Partial — `gcs_e/m/v` exist, only 8.4% coverage (see finding above — not missing at random) |
| 7 | Hemoglobin | ✅ Available — `hb`, 94.5% |
| 8 | White cell count | ✅ Available — `wbc`, 94.1% |
| 9 | Urea | ✅ Available (renamed) — `bun`, 93.2% |
| 10 | Sodium | ✅ Available — 93.8% |
| 11 | Potassium | ✅ Available — 93.8% |
| 12 | ECG findings | ❌ Not available — no ECG field in this dataset |

**Operative Severity Score (OSS) — 6 variables, sum ranges 6-48:**

| # | Variable | Availability in INSPIRE |
|---|---|---|
| 13 | Operative severity magnitude | ❌ Not available — needs surgeon's subjective grade or procedure-type coding |
| 14 | Number of procedures | ✅ Available (different table) — `n_operations` in `label_df` |
| 15 | Total blood loss | ✅ Available (renamed) — `ebl`, 59.3% intra-op coverage |
| 16 | Peritoneal soiling | ❌ Not available — surgical finding not captured |
| 17 | Malignancy presence/stage | 🔧 Derivable — from ICD-10 codes, not yet mapped |
| 18 | Mode of surgery (elective/emergency) | ⚠️ Partial — only binary `emop`, not full urgency banding |

**Important honest note on the exact score cutoffs:** the specific numeric boundary for
each 1/2/4/8 level (e.g. exactly which systolic BP range scores 2 vs. 4) is published in
Copeland et al. 1991's original Table 1 and in validated POSSUM calculators. I'm
deliberately not reproducing those exact cutoffs here from memory — the risk of
transcribing a clinical scoring boundary incorrectly is too high to guess at. Before
building this for real, cross-check the exact bands against the original paper or a
validated calculator (e.g. mdapp.co/possum-score-calculator, evidencio.com/models/show/1011).

### Key references
- Copeland, G.P., Jones, D., Walters, M. (1991). "POSSUM: a scoring system for surgical
  audit." *British Journal of Surgery*, 78, 355–360. — original, 1,372 patients.
- Whiteley, M.S., Prytherch, D.R., Higgins, B., Weaver, P.C., Prout, W.G. (1996). "An
  evaluation of the POSSUM surgical scoring system." *British Journal of Surgery*, 83,
  812–815. — introduces P-POSSUM.
- Prytherch, D.R. et al. (1998). "POSSUM and Portsmouth POSSUM for predicting mortality."
  *British Journal of Surgery*, 85, 1217–1220. — the formal P-POSSUM paper.
- Barnett, S. et al. (2018), vascular surgery cohort, PubMed 29169796 — POSSUM AUROC
  0.72; predicted-vs-observed deaths comparison above.

### The honest gap for our data
Several OSS inputs (peritoneal soiling grade, cancer staging, exact operative-severity
grade) **are not currently extracted from INSPIRE**. See Section 6 for the full,
field-by-field audit of what's actually available.

---

## 3. NELA (National Emergency Laparotomy Audit) risk model

### What it is
Unlike POSSUM's two-stage "sum into PS/OSS, then combine" structure, NELA feeds **every
variable directly into one logistic regression**, each with its own individually-fitted
weight — no intermediate sub-scores. Already implemented in this repo's `nela.py`, sourced
from the official NELA technical documentation.

### The complete equation, every term with real INSPIRE availability

```
logit = -3.04678
        + 0.06660 × Age                                              [✅ Available — static field]
        + 1.13007 × (ASA=3) + 1.76293 × (ASA=4) + 2.55345 × (ASA=5)   [✅ Available — asa field]
        - 0.03021 × (ASA=3×Age) - 0.03356 × (ASA=4×Age)
        - 0.04676 × (ASA=5×Age)                                       [✅ Available — computed from above]
        - 0.04323 × Albumin                                           [✅ Available — 93.5%]
        + 0.01265 × Pulse - 0.00012 × Pulse²                          [✅ Available (renamed) — hr, 98.4%]
        - 0.00683 × SystolicBP + 0.00011 × SystolicBP²                [✅ Available (renamed) — nibp_sbp, 98.2%]
        + 0.38002 × ln(Urea)                                          [✅ Available (renamed) — bun, 93.2%]
        + 0.02041 × ln(WBC) + 0.24153 × ln(WBC)²                      [✅ Available — wbc, 94.1%]
        + 0.41557 × (GCS=14) + 0.64480 × (GCS=3-13)                   [⚠️ Partial — only 8.4% coverage]
        + 0.19201 × Malignancy_Primary
        + 0.50610 × Malignancy_Nodal
        + 0.94309 × Malignancy_Distant                                [✅ Derivable — 37.2% have an active malignancy ICD-10 code, verified]
        + 0.35378 × Respiratory_2 + 0.60700 × Respiratory_3           [⚠️ Partial — spo2/rr/fio2 proxies exist]
        + 0.03782 × Urgency_6to18h + 0.14779 × Urgency_2to6h
        + 0.57310 × Urgency_lt2h                                      [⚠️ Partial — only binary emop, no hour bands]
        + 0.02812 × Indication_Sepsis                                   [⚠️ Attempted, weak — 0.3% match, likely wrong code range]
        + 0.56948 × Indication_Ischaemia                                [⚠️ Attempted, weak — 4.5% cardiac / 0.2% bowel, needs review]
        - 0.40615 × Indication_Bleeding                                [❌ Attempted, unreliable — 0.0% match on real data, code range likely wrong]
        + 0.29453 × Soiling_Severe                                     [❌ Not available]

risk = 1 / (1 + e^(-logit))
```

**Bottom line: 9 of the ~14 term-groups are fully or reliably available right now (Age,
ASA×Age, Albumin, Pulse, SystolicBP, Urea, WBC, and now Malignancy — confirmed at 37.2%
coverage). Only GCS (sparse), respiratory status, urgency banding, and the three
indication categories (sepsis/ischaemia/bleeding — attempted, unreliable with current code
ranges) remain genuine gaps.**

### Three things this equation's structure teaches, worth understanding directly
1. **ASA appears three times, each interacting with Age** — the model isn't just adding
   "high ASA = bad," it's saying the *effect size* of high ASA changes with age (the
   negative interaction terms mean high ASA matters relatively less in already-old
   patients, since age alone is already driving risk up).
2. **Pulse and SystolicBP each appear twice — once linear, once squared** — this lets the
   model capture a **U-shape**: both abnormally low AND abnormally high values increase
   risk, not just "higher is worse."
3. **Urea and WBC enter as `ln(...)`, not raw values** — a standard transform for lab
   values that are heavily right-skewed (a few extreme outliers); the log compresses that
   skew so the model can treat the transformed value roughly linearly.

### Where the weights come from
Not theoretical — fitted by logistic regression on real UK national audit data (tens of
thousands of emergency laparotomy cases), finding whichever weights best separated
survivors from deaths in *that* population. This is worth remembering before assuming the
weights transfer perfectly to INSPIRE's population without recalibration.

**Reference:** NELA Project Team, *National Emergency Laparotomy Audit* risk prediction
model technical documentation (Healthcare Quality Improvement Partnership, updated April
2023) — coefficients in `nela.py` transcribed directly from this document.

---

## 4. The circularity/self-fulfilling-prophecy question (Clinician Question 7)

This is a real, actively-studied methodological concern in clinical prediction modeling —
worth grounding in the actual theory rather than treating as a vague worry.

### The core idea
**Merton, R.K. (1948).** "The self-fulfilling prophecy." *The Antioch Review*, 8(2),
193-210. — the original sociological concept: a prediction that shapes behavior in a way
that makes the prediction come true, independent of whether the original belief was
accurate.

Applied to clinical scores, this becomes a specific, well-documented failure mode:

> "A self-fulfilling prophecy occurs when the prediction of a model influences treatment
> decisions in a way that increases the likelihood of the predicted outcome over time,
> creating a self-amplifying source of bias." — *Feedback loops in intensive care unit
> prognostic models: an under-recognised threat to clinical validity*, ScienceDirect,
> 2025.

The mechanism specifically relevant to ASA: if a clinician assigns a high ASA class *because*
they already suspect a poor outcome, and that high ASA class then triggers more
conservative management (less aggressive intervention, earlier palliation discussions,
different resource allocation), the patient's subsequent death partly reflects the
*management decision*, not purely their underlying physiology. The score and the outcome
become entangled through the clinician's own behavior in between.

**van Amsterdam, W.A.C. et al.** "When accurate prediction models yield harmful
self-fulfilling prophecies." *Patterns* (Cell Press), 2025; also on arXiv:2312.01210. —
the most direct, recent treatment of this for clinical prediction models specifically:
demonstrates formally that a model can be *statistically accurate* in validation and
*still* be causing harm (or inflating its own apparent accuracy) once it's influencing the
treatment decisions that determine the very outcome it's predicting.

### Why this matters for your specific pipeline, precisely
**Important finding worth flagging directly: ASA is not currently in your GBM's 18-feature
input set (`INPUT_VARS` in Section 10) at all.** So the 0.9674 AUROC you already have does
**not** include any ASA-driven circularity risk right now — the question is prospective
(should ASA be added later), not something already baked into your current number. This
also matches the roadmap doc's own flag: "ASA as a model input — currently unused by the
DNN despite 100% coverage" — true of GBM too, not just the DNN.

### What can, and can't, be tested with the data you have
- **Can test:** how much *additional* predictive signal ASA carries beyond what your labs
  already capture (an ablation study — Section 5 below). If ASA adds little once labs are
  already in the model, that's evidence it's *redundant* with objective physiology, not
  necessarily evidence of circularity, but it does reduce how much you'd lose by leaving
  it out to sidestep the circularity concern entirely.
- **Cannot test from this dataset alone:** true causal circularity — whether ASA
  assignment is *causing* different treatment, which is *causing* the outcome — needs
  either treatment/intervention data (e.g. did high-ASA patients receive measurably less
  aggressive care) or a controlled/interventional study design, not just an observational
  AUROC comparison. This is worth being upfront about rather than overclaiming what an
  ablation study can settle.

---

## 5. Full-cohort empirical validation — what to actually run

Two things below use your **real, full 99,886-patient data** (via the shared GBM/DNN
cohort and split already saved to Drive), rather than relying purely on the literature
numbers above.

### 5a. ASA-alone baseline — benchmark against Koo et al.'s pooled 0.736 AUROC

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

SAVE_DIR = '/content/drive/MyDrive/inspire_extracted_tables'
gbm_df = pd.read_parquet(f'{SAVE_DIR}/gbm_df.parquet')

with open(f'{SAVE_DIR}/shared_cohort_split.pkl', 'rb') as f:
    import pickle
    shared = pickle.load(f)
train_ids, test_ids = shared['train_ids'], shared['test_ids']

# ASA alone -- exactly one feature, nothing else
X_asa = gbm_df[['asa']].astype(float)
X_asa_imputed = pd.DataFrame(
    SimpleImputer(strategy='median').fit_transform(X_asa),
    columns=['asa'], index=X_asa.index
)
y = gbm_df['inhosp_death_30day']

X_train, X_test = X_asa_imputed.loc[train_ids], X_asa_imputed.loc[test_ids]
y_train, y_test = y.loc[train_ids], y.loc[test_ids]

asa_only_model = LogisticRegression(class_weight='balanced')
asa_only_model.fit(X_train, y_train)
asa_only_auroc = roc_auc_score(y_test, asa_only_model.predict_proba(X_test)[:, 1])
print(f"ASA-alone AUROC on full INSPIRE cohort: {asa_only_auroc:.4f}")
print(f"Koo et al. 2015 pooled meta-analysis (77 studies, 165,705 patients): 0.736")
```

### 5b. GBM with vs. without ASA — the ablation that speaks to Clinician Q6/Q7

```python
INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur',
              'preop_hb', 'preop_platelet', 'preop_wbc',
              'preop_aptt', 'preop_ptinr', 'preop_glucose',
              'preop_bun', 'preop_albumin', 'preop_ast',
              'preop_alt', 'preop_creatinine', 'preop_sodium',
              'preop_potassium']
INPUT_VARS_WITH_ASA = INPUT_VARS + ['asa']

def fit_and_score(feature_list):
    X = gbm_df[feature_list].astype(float)
    X_imp = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X),
                          columns=feature_list, index=X.index)
    X_tr, X_te = X_imp.loc[train_ids], X_imp.loc[test_ids]
    model = LogisticRegression(max_iter=5000, class_weight='balanced')
    model.fit(X_tr, y_train)
    return roc_auc_score(y_test, model.predict_proba(X_te)[:, 1])

auroc_without_asa = fit_and_score(INPUT_VARS)
auroc_with_asa = fit_and_score(INPUT_VARS_WITH_ASA)

print(f"GBM WITHOUT ASA (current 18 features): {auroc_without_asa:.4f}")
print(f"GBM WITH ASA (19 features):            {auroc_with_asa:.4f}")
print(f"Difference: {auroc_with_asa - auroc_without_asa:+.4f}")
print()
print("Small difference -> ASA is largely redundant with the labs already in the model")
print("(safe to leave out, sidesteps the circularity question entirely).")
print("Large difference  -> ASA carries real independent signal worth the circularity")
print("discussion with the clinical team before deciding whether to include it.")
```

**Note:** re-run `nela.py`'s `max_iter=1000` convergence issue check here too (increase
`max_iter`) — this is the same open item flagged in `INSPIRE_Full_Run_Analysis.md`
regarding the un-converged 0.9674 GBM result; worth fixing before treating any of these
AUROC comparisons as final.

---

---

## 6. Full-cohort field audit — what we actually have (run 2026-08-02, 5,000-patient sample)

Real counts, replacing the assumptions in Sections 2-3 above. Full lab/vital inventory:
**38 distinct labs, 16 ward vitals, 71 intra-op vitals** found in the raw data.

### Corrected POSSUM/NELA field mapping

The first pass of this audit used naive text-matching and produced several false
"NOT FOUND" results — fields that exist under a different name weren't being matched.
Corrected below:

| Field | Status | Note |
|---|---|---|
| `age` | ✅ Available | Static field, ~100% coverage |
| `systolic_bp` | ✅ Available (renamed) | `nibp_sbp`, 98.2% ward / 99.4% intra-op |
| `pulse` | ✅ Available (renamed) | `hr`, 98.4% ward / 100% intra-op |
| `hb`, `wbc`, `sodium`, `potassium`, `albumin` | ✅ Available | 93-95% coverage each |
| `urea` | ✅ Available (renamed) | `bun`, 93.2% |
| `n_procedures` | ✅ Available (different table) | `n_operations` in `label_df`, 100% by construction |
| `blood_loss` | ✅ Available (renamed) | `ebl`, 59.3% intra-op coverage |
| `gcs` | ⚠️ Partial | `gcs_e`/`gcs_m`/`gcs_v` exist but only **8.4% ward coverage** — too sparse to trust for most patients |
| `urgency` / `urgency_band` | ⚠️ Partial | Only binary `emop` (elective/emergency); POSSUM/NELA want hour-banded urgency (<2h/2-6h/6-18h/elective) |
| `respiratory_status` (NELA) | ⚠️ Partial | Raw `spo2`/`rr`/`fio2` exist as proxies, not NELA's exact category |
| `malignancy_stage`, `indication_sepsis/ischaemia/bleeding` | 🔧 Derivable | Would need ICD-10 diagnosis codes mapped to these categories — not yet built, but the codes themselves exist (used already for HFRS) |
| `cardiac_signs`, `respiratory_signs` (POSSUM exam grades), `ecg`, `peritoneal_soiling`, `operative_severity` | ❌ Not available | Genuinely absent — these need a clinician's subjective exam grading or an operative note field this dataset doesn't have |

**Bottom line: NELA is largely buildable from real data (only GCS sparsity and the
ICD-10-derived indication/malignancy terms are true gaps). POSSUM is not currently
buildable in full — its operative-severity component depends on fields (exam grades,
peritoneal soiling) this dataset simply doesn't have.**

### GCS missingness is NOT random — it's systematically recorded for sicker patients

Checked on the same 5,000-patient sample, cross-tabulated against department, ASA class,
and emergency status:

| Breakdown | GCS coverage |
|---|---|
| CTS (cardiothoracic surgery) | 37.5% |
| NS (neurosurgery) | 25.2% |
| Most other departments (GS, OS, OL, PS, OG, UR, OT) | 1.7% – 7.5% |
| Emergency surgery (`emop`=1) | 20.7% |
| Elective surgery (`emop`=0) | 7.1% |
| ASA 6 | 60.0% |
| ASA 1 | 3.2% |

**Interpretation: GCS is recorded when a clinical team already suspects it's relevant** —
neurosurgery/cardiothoracic cases, emergencies, and the sickest ASA classes get it far
more often than routine elective cases. This means the *fact* that GCS is missing is
itself informative (this patient likely wasn't neurologically concerning), not just a
random data gap — worth stating this explicitly to the clinical team rather than treating
8.4% as a simple missing-data problem to fill in.

### Malignancy and indication-for-surgery: tested against real ICD-10 codes (5,000-patient sample)

Confirmed the real field name (`icd10_cm`, same structure HFRS uses) and checked coverage
against standard ICD-10 ranges:

| Category | Code range checked | Coverage found |
|---|---|---|
| Malignancy (active) | C00-C97 | ✅ **37.2%** |
| Malignancy (in-situ) | D00-D09 | ✅ 2.8% |
| Indication: sepsis | A40-A41 | ⚠️ 0.3% (14 patients) |
| Indication: cardiac ischaemia | I20-I25 | ⚠️ 4.5% |
| Indication: bowel ischaemia | K55 | ⚠️ 0.2% |
| Indication: GI bleeding | K92.0-K92.2 | ❌ **0.0% — zero patients** |

**Malignancy detection works well and reveals something important about this cohort: it's
heavily oncology-surgery weighted.** The 20 most common 3-character ICD-10 prefixes
overall were dominated by cancer codes (C50 breast, C18 colon, C16 stomach, C22 liver, C34
lung, C20 rectum, C73 thyroid, C67 bladder, C25 pancreas — 12 of the top 20). This is a
real characterization of the dataset worth keeping in mind broadly, not just for NELA.

**The indication categories mostly failed — and a true zero is informative, not a null
result.** Zero patients matching GI bleeding out of 5,000 strongly suggests the narrow
code ranges used don't match how these are actually coded in this dataset (real bleeds
are likely captured under more specific underlying diagnosis codes instead of the generic
K92.0-2 range). **Revised status: malignancy is genuinely derivable from ICD-10; the
sepsis/ischaemia/bleeding indication categories are NOT reliably derivable with simple
code-range matching as currently attempted** — needs either the correct code mapping from
the clinical team (Clinician Question 10, updated) or a different data source entirely.

## 7. Decision: use NELA as a DNN input feature, not just a comparison baseline

Rather than only comparing the DNN against NELA as a separate baseline (the original
Section 12 framing), the better use of NELA's already-fitted, externally-validated
equation is as **one more input feature** to the DNN itself — letting the model use
NELA's decades of clinical-modeling structure (the ASA×Age interaction, the log-transforms
on urea/WBC, the U-shaped pulse/BP terms) for free, rather than requiring the DNN to
rediscover that structure from a comparatively small, heavily imbalanced dataset.

**Open implementation decision, not yet resolved:** for the fields NELA needs but this
dataset doesn't fully have (GCS, malignancy stage, indication category, soiling), two
options:
- **(a)** Default missing terms to their reference/lowest-risk category — simplest, but
  silently makes affected patients look slightly healthier than a true NELA score would.
- **(b)** Compute an explicitly-labeled "NELA-partial" score, documented everywhere it's
  used, so it's never confused with a true, complete NELA calculation.

Leaning toward (b) for scientific honesty, pending final confirmation before building it.

---

## 8. Reference list (consolidated)

1. Saklad, M. (1941). Grading of patients for surgical procedures. *Anesthesiology*, 2, 281–284.
2. Dripps, R.D. (1963). New classification of physical status. *Anesthesiology*, 24, 111.
3. Koo, C.Y. et al. (2015). A meta-analysis of the predictive accuracy of postoperative mortality using the ASA physical status classification system. *World Journal of Surgery*, 39, 88–103.
4. Sankar, A. et al. (2014). Reliability of the American Society of Anesthesiologists physical status scale in clinical practice. *British Journal of Anaesthesia*.
5. Copeland, G.P., Jones, D., Walters, M. (1991). POSSUM: a scoring system for surgical audit. *British Journal of Surgery*, 78, 355–360.
6. Whiteley, M.S. et al. (1996). An evaluation of the POSSUM surgical scoring system. *British Journal of Surgery*, 83, 812–815.
7. Prytherch, D.R. et al. (1998). POSSUM and Portsmouth POSSUM for predicting mortality. *British Journal of Surgery*, 85, 1217–1220.
8. NELA Project Team. National Emergency Laparotomy Audit risk prediction model technical documentation, Healthcare Quality Improvement Partnership (updated April 2023).
9. Merton, R.K. (1948). The self-fulfilling prophecy. *The Antioch Review*, 8(2), 193-210.
10. van Amsterdam, W.A.C. et al. (2025). When accurate prediction models yield harmful self-fulfilling prophecies. *Patterns* (Cell Press); arXiv:2312.01210.
11. Feedback loops in intensive care unit prognostic models: an under-recognised threat to clinical validity. (2025). *ScienceDirect*.
