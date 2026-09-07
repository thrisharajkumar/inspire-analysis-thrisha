# INSPIRE Pipeline — Full-Scale Run: What We Did, What We Found, What It Means

This document explains, in plain language, what each part of the notebook run actually
tested, what came out of it, and how those findings feed into your broader research
questions (class-imbalance handling, multi-modal DNN pipeline).

**Cohort:** 99,886 patients total (98,944 "survived", 942 "died" by folder label).

---

## 1. Why we ran this at all

The original notebook was built and validated on a 30-patient toy subset. Before trusting
*any* number from it — a mortality rate, an AUROC, a feature importance — we needed to
confirm the whole pipeline actually works, and produces trustworthy numbers, on the real
99,886-patient cohort. That's what this run was for: not to finalize a model, but to get
the first real, full-scale numbers on the table.

---

## 2. Is the mortality *label* even correct? (Section 3)

**What we checked:** the dataset organizes patients into `survived/` and `died/` folders.
We tested whether that folder name agrees with a proper clinical definition — died within
30 days of their (last) operation.

**What we found:**
- Folder label vs. "died at any point" — **100.0% match**
- Folder label vs. "died within 30 days of operation" — **99.5% match**
- **473 patients** are folder-labeled "died" but did **not** die within 30 days of their
  operation — meaning they died later, of something else, or their death wasn't clearly
  tied to the surgery.

**In simple terms:** the folder names are a good label almost all the time, but not
perfect. Using the folder name directly (as the original 30-patient demo did) would
mislabel 473 patients out of ~99,886 — small in percentage (0.5%), but real: **469 died
within 30 days vs. 942 in the "died" folder** — the true 30-day death count is roughly
**half** what the folder name implies.

**Takeaway for your research:** always use the corrected `died_30day` definition, not the
raw folder name, for any model training or reporting. **True cohort mortality: 469 died /
99,417 survived — 0.5%.** This number is central to your class-imbalance work: you're
working with roughly a **200:1** imbalance (`pos_weight ≈ 212`), not the 50:50 you'd get by
naive folder-based labeling.

---

## 3. Does surgery *order* matter for patients with multiple operations? (Section 4)

**What we checked:** 21,565 of the 99,886 patients (~22%) had more than one operation.
For these patients, should "died within 30 days" be measured from their *first* operation
or their *last* one? The current pipeline uses the last operation.

**What we found:**
- **111 patients** get a *different* mortality label depending on which operation you
  measure from.
- Patients with multiple operations have a **higher** 30-day mortality rate (0.8%) than
  single-operation patients (0.4%) — twice as high.

**In simple terms:** most of the time it doesn't matter which operation you anchor to, but
for a small group it does, and multi-operation patients are meaningfully sicker as a
group.

**Takeaway for your research:** don't casually drop multi-op patients to "simplify" the
data — they're a higher-risk subgroup, and dropping them would bias your model toward
easier cases. If anything, this is a signal worth its own feature (e.g. "had a prior
operation" as a risk factor), not a group to exclude.

---

## 4. Does ASA class behave the way it clinically should? (Section 5)

**What we checked:** ASA is a standard 1-6 anesthesia risk classification, assigned by
the clinical team before surgery. If the data is trustworthy, mortality should climb
steadily with ASA class.

**What we found:**

| ASA class | Mortality rate | Patients |
|---|---|---|
| 1 | 0.07% | 34,748 |
| 2 | 0.17% | 54,107 |
| 3 | 2.16% | 8,024 |
| 4 | 10.3% | 464 |
| 5 | 24.2% | 33 |
| 6 | 82.5% | 57 |

**In simple terms:** this is a clean, expected staircase — mortality roughly triples or
more with each step up in ASA class, up to a dramatic 82.5% mortality at class 6 (which is
reserved for brain-dead organ donors, so a near-certain "death" outcome there is actually
*expected*, not alarming).

**Takeaway for your research:** this is a strong sanity check that the underlying dataset
is clinically coherent, not corrupted or scrambled. ASA is also, on its own, a
surprisingly powerful single predictor — worth keeping as a strong baseline feature or
even a baseline model to beat.

---

## 5. Which features actually predict mortality, and which are noise? (Sections 6-7)

**What we checked:** ran a statistical screen (54 candidate lab/vital features) to find
which ones are genuinely associated with mortality, correcting for the fact that testing
54 features at once creates false positives by chance (FDR correction), then further
checked which ones survive after accounting for department (since some departments simply
have sicker patients).

**What we found:**
- **46 of 54** features passed the initial significance screen.
- After removing redundant/duplicate features (like `alt` overlapping with `ast`), **28**
  remained.
- After controlling for department, **15 features** are independently predictive,
  including: **albumin, platelets, hemoglobin, heart rate, BUN, creatinine, glucose,
  lymphocyte count, CRP** — all clinically sensible markers of physiological reserve,
  organ function, and inflammation.

**Takeaway for your research:** this 15-feature list is a strong candidate for your GBM
model's input set, and a good starting point for deciding which lab channels matter most
if you're choosing a reduced feature set for the DNN. Notably, low albumin and low
hemoglobin (markers of malnutrition/frailty and anemia) were the two strongest predictors
— consistent with frailty being a major mortality driver, which connects directly to the
next section.

---

## 6. Does frailty (HFRS) predict mortality independent of diagnoses? (Section 8)

**What we checked:** HFRS (Hospital Frailty Risk Score) is computed from a patient's
diagnosis codes and buckets them into frailty categories.

**What we found:**

| HFRS category | Mortality rate | Patients |
|---|---|---|
| High | 1.01% | 6,024 |
| Intermediate | 0.77% | 8,457 |
| Low | 0.41% | 80,865 |
| Unknown (no diagnoses) | 0.18% | 4,540 |

**In simple terms:** frailty category is a real, ordered signal — high-frailty patients
die at roughly 5-6x the rate of low-frailty patients. But note this HFRS-based signal is
weaker than ASA's (ASA showed up to 82% mortality at the extreme, HFRS tops out around 1%)
— HFRS captures a real but comparatively modest slice of risk on its own.

**Takeaway for your research:** frailty is a legitimate, independent-feeling signal, but
not a dominant one by itself — it likely adds value as one input among many rather than
being a strong standalone predictor. Worth including as a feature, not worth over-relying
on.

---

## 7. Clinical baseline scores (NELA, POSSUM, NEWS2) — Section 9

**Important:** these were **not run on real patients**. They were demo calls with made-up
illustrative input values, just to confirm the scoring equations themselves work
correctly. **No real NELA/POSSUM/NEWS2 score exists yet for any actual INSPIRE patient.**

**Takeaway for your research:** if you want per-patient NELA/POSSUM scores as a comparison
baseline, that mapping (INSPIRE fields → NELA/POSSUM input variables, computed for all
99,886 patients) still needs to be built — it's a real gap, not a completed baseline.

---

## 8. The two real models — GBM vs. DNN (Sections 10-12)

This is the core comparison, and it needs the most caveats.

### GBM baseline (Logistic Regression, 18 pre-op features)
**Result: AUROC = 0.9674**, trained/tested on the full shared cohort (65,952 train /
32,484 test patients, same population both models were meant to use).

**⚠️ This number needs a sanity check before you trust it.** Two reasons:
1. It's *higher* than the published benchmark you're comparing against (Shickel et al.,
   0.92, on 56,242 patients) and higher than most published surgical mortality models.
   That's not impossible, but it's unusual enough to be suspicious rather than simply
   celebrated.
2. The model **did not fully converge** — scikit-learn raised a `ConvergenceWarning`
   ("lbfgs failed to converge after 1000 iterations"). The reported 0.9674 may not even be
   the model's true optimum; it could shift if you increase `max_iter` or scale the
   features first.

**Before reporting 0.9674 anywhere, two things should be checked:** (a) re-run with a
higher `max_iter` (e.g. 5000) and/or scaled features to see if the number holds, and (b)
check the 18 input features for anything that could be indirectly leaking the outcome
(e.g. a lab value only drawn *because* the patient was critically ill, which is a subtly
different thing from a lab value that predicts future risk).

### DNN transformer (two-phase autoencoder + classifier)
**Result: AUROC = 0.7525** — but this is **not the real result**. The full-scale DNN run
crashed twice from memory exhaustion, even after several fixes. This number comes from a
small fallback sanity-check run (800 train / 300 test patients), with only **5 positive
(died) cases** in the entire test set — far too few to trust statistically.

**Takeaway for your research:** you do not yet have a real, trustworthy DNN result to
compare against GBM. The 0.75 vs. 0.97 comparison in the results table is **not a fair
fight** — one number is real and one is a rough sanity check on a tiny, statistically
underpowered sample. Don't draw "GBM beats DNN" conclusions from this yet.

---

## 9. What's still genuinely unfinished (Section 13)

- **13a (organ-system feature grouping):** just a static dictionary grouping features by
  body system (renal, cardiovascular, respiratory, etc.) — not yet wired into any model.
  This is exactly the scaffold your multi-modal DNN idea would build on.
- **13b (attention audit):** empty — deliberately left as a TODO, no code exists yet.
- **13c (calibration check):** not yet run against the real full-scale GBM model.

---

## Summary: What you actually know now, vs. what you don't yet

**Solid, trustworthy findings:**
- True 30-day mortality is **0.5%** (469/99,886) — not the ~1% the folder names implied.
- ASA class and HFRS frailty category are both real, clinically coherent risk signals.
- 15 lab/vital features are statistically independent predictors after confound
  adjustment — a strong candidate feature set.
- Multi-op patients are a higher-risk subgroup (0.8% vs 0.4% mortality) and shouldn't be
  dropped from analysis.

**Not yet trustworthy / not yet done:**
- GBM's 0.9674 AUROC needs a convergence + leakage check before you rely on it.
- DNN has no real full-scale result yet — only a small, low-power sanity check.
- NELA/POSSUM have no real per-patient scores yet, only demo equations.
- The multi-modal organ-system architecture is a dictionary, not a working model yet.

---

## Direct relevance to your next two research threads

**Class imbalance sampling techniques:** your real working number is **469 deaths out of
99,886 patients (0.5% mortality, ~212:1 imbalance)**. This is severe enough that
`class_weight='balanced'` / `pos_weight` reweighting alone (what's currently implemented)
is a reasonable start but likely insufficient on its own — this is a good candidate
dataset to test SMOTE-style oversampling, focal loss, or threshold-tuned evaluation
against the current reweighting-only baseline, and to report **AUPRC alongside AUROC**,
since AUROC can look deceptively strong under this level of imbalance (worth checking
whether GBM's 0.9674 AUROC comes with a much less impressive AUPRC — that comparison
alone might partly explain the suspiciously high AUROC).

**Multi-modal DNN pipeline:** the 15 confound-adjusted features from Section 6-7 and the
organ-system grouping dictionary from Section 13a are your natural starting inputs — the
statistical groundwork for "which signals matter" is already done; what's missing is
architecture that treats different organ systems as separate input streams before fusing
them, rather than the current single flat 7-feature time series. The DNN's actual
full-scale training also needs to succeed before any multi-modal extension is worth
building on top of it.
