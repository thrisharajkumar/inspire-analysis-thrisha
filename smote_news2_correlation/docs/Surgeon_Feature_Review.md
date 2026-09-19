# Feature Inventory for Clinical Review

**Purpose of this document:** every physiological measurement, lab value, and
medication/device flag currently used by the mortality model, organized by organ
system. Please review each system's table and flag anything that looks clinically
wrong, incomplete, or mis-assigned -- your input here directly shapes the model.

**What "SOFA-analogous role" means:** for intra-operative drug doses and device
settings (vasopressors, ventilator settings, transfusions), we assign them to an organ
system using the same logic as the SOFA score (a standard ICU severity score) --
e.g. a vasopressor dose counts as a cardiovascular signal, since needing that support
*is* a marker of cardiovascular compromise, not just a side detail.

**Three time windows:** Pre-op (admission to surgery start), Peri-op (during surgery),
Post-op (surgery end to discharge/30-day window). A feature may apply in more than one.

---

## 1. Renal

| Feature | What it is | When measured | Role |
|---|---|---|---|
| creatinine | Kidney function marker | Pre-op, Post-op | Core renal marker |
| bun | Blood urea nitrogen, kidney function | Pre-op, Post-op | Core renal marker |
| sodium, potassium, chloride | Electrolytes | Pre-op, Post-op | Electrolyte/renal function |
| calcium, phosphorus, ionized calcium | Mineral levels | Pre-op, Post-op | Renal-mineral axis |
| crrt | Continuous renal replacement therapy (dialysis) in use | Pre-op, Post-op | Renal organ **support** (needing dialysis) |
| urine output | Direct kidney output | Pre-op (rare), during surgery, Post-op | Direct renal function signal |
| hydroxyethyl starch (fluid) | IV volume expander | During surgery | **Flagged for your input**: known renal toxicity risk -- kept visible here as a caveat, not treated as a primary renal signal |

**Question for you:** does the hydroxyethyl starch flag belong here, or should it move
elsewhere / be dropped as too indirect a signal?

---

## 2. Cardiovascular

| Feature | What it is | When measured | Role |
|---|---|---|---|
| troponin I/T, CK, CK-MB | Cardiac injury markers | Pre-op, Post-op | Cardiac injury markers |
| heart rate, blood pressure (non-invasive) | Core hemodynamics | Pre-op, Post-op | Core hemodynamics |
| arterial BP, heart rate | Continuous monitoring | During surgery | Continuous intra-op hemodynamics |
| cardiac index, central venous pressure, stroke volume index | Cardiac performance | During surgery | Cardiac performance/preload |
| pulmonary artery pressure | Right-heart pressure | During surgery | Right-heart/pulmonary pressure |
| ST-segment leads (I, II, III, V5) | Ischemia monitoring | During surgery | Ischemia monitoring |
| dobutamine, dopamine, milrinone (inotrope doses) | Heart-strengthening drugs | During surgery | Organ **support** signal |
| epinephrine, norepinephrine, phenylephrine, vasopressin, ephedrine (vasopressor doses) | Blood-pressure-raising drugs | During surgery | Organ **support** signal |
| nitroglycerin (vasodilator) | Blood-pressure-lowering drug | During surgery | Organ **support**, opposite direction |
| IABP | Mechanical heart support device in use | Pre-op, Post-op | Mechanical cardiac support flag |

**Existing hand-coded link:** cardiovascular summary stats (mean HR, MAP deviation,
IABP flag) are currently fed directly into the renal branch, encoding the known
cardiorenal relationship by hand. **New addition below (§ System Correlation Layer)**
lets the model discover other cross-system relationships too, rather than only the one
we hard-coded.

---

## 3. Respiratory

| Feature | What it is | When measured | Role |
|---|---|---|---|
| pao2, paco2, pH, bicarbonate, base excess, sao2 | Blood gas | Pre-op, Post-op | Blood gas / respiratory function |
| SpO2, respiratory rate, FiO2 | Core respiratory monitoring | Pre-op, during surgery, Post-op | Core respiratory monitoring |
| medical air, O2 flow, N2O, end-tidal CO2 | Ventilation/gas exchange | During surgery | Ventilation/gas exchange |
| minute volume, PEEP, peak/plateau/mean airway pressure, tidal volume | Ventilator settings | During surgery | Organ **support** signal |
| anesthetic gas concentrations (desflurane, isoflurane, sevoflurane, etc.) | Anesthesia delivery | During surgery | Respiratory-delivered anesthetic |
| ventilator, ECMO | Mechanical respiratory support in use | Pre-op, Post-op | Mechanical support flag |

---

## 4. Metabolic / Hepatic

| Feature | What it is | When measured | Role |
|---|---|---|---|
| glucose, HbA1c | Blood sugar control | Pre-op, Post-op | Metabolic control |
| albumin, ALP, ALT, AST, bilirubin, total protein | Liver function | Pre-op, Post-op | Hepatic function |
| lactate | Tissue perfusion/stress marker | Pre-op, Post-op | Perfusion/metabolic stress |
| body temperature | Thermoregulation | Pre-op, during surgery, Post-op | Thermoregulation |
| dextrose infusions (D5W, D10W, D50W) | Sugar-water IV fluids | During surgery | Metabolic support |
| albumin infusions | Protein replacement | During surgery | Hepatic-synthetic-function-adjacent |

---

## 5. Haematology / Coagulation

| Feature | What it is | When measured | Role |
|---|---|---|---|
| hemoglobin, hematocrit, WBC, platelets | Core blood counts | Pre-op, Post-op | Core haematology |
| aPTT, PT/INR, fibrinogen, D-dimer | Clotting function | Pre-op, Post-op | Coagulation |
| CRP, lymphocytes, segmented neutrophils | Inflammatory/immune | Pre-op, Post-op | Inflammatory/immune |
| estimated blood loss | Direct surgical blood loss | During surgery | Direct haematologic stress event |
| red cells, fresh frozen plasma, platelets, cryoprecipitate, pheresis products | Transfusion given | During surgery | Organ **support** signal |

---

## 6. Neurological

| Feature | What it is | When measured | Role |
|---|---|---|---|
| Glasgow Coma Scale (eye/motor/verbal) | Consciousness level | Pre-op, Post-op | Core neuro function |
| BIS (bispectral index) | Anesthesia depth monitor | During surgery | Consciousness-level proxy |
| cerebral regional oxygen saturation | Brain oxygenation | During surgery | Cerebral oxygenation |
| propofol, midazolam, fentanyl, sufentanil, remifentanil, alfentanil | Sedatives/opioids | During surgery | Modulates/proxies consciousness during surgery |

---

## 7. Diagnosis-derived (not time-varying)

| System | Source | Notes |
|---|---|---|
| Gastrointestinal (GI) | ICD-10 diagnosis chapter XI | Static flag, not a measurement over time |
| Musculoskeletal (MSK) | ICD-10 diagnosis chapter XIII | Static flag, not a measurement over time |

---

## 8. Fluid / Resuscitation -- flagged as an open question, not yet a settled decision

| Feature | What it is | Why it's separate |
|---|---|---|
| normal saline, half-normal saline, Hartmann's solution, plasma solution | IV crystalloid fluids given | Reflects a **clinician's decision** during surgery more than a patient's own organ state -- currently NOT assigned to any organ system, kept as a separate aggregate total instead |

**Question for you:** does total fluid volume belong in the model at all, or is it too
confounded with surgery duration/blood loss to be a genuine risk signal rather than
just a proxy for "this was a longer/bloodier operation"?

---

## What's new in this update, specifically for your review

1. **NEWS2 score** -- the standard UK early-warning score (Royal College of
   Physicians), computed from respiratory rate, SpO2, oxygen use, temperature,
   systolic BP, heart rate, and consciousness level -- all already collected above.
   One approximation worth your input: NEWS2 normally uses an AVPU consciousness
   assessment, which isn't directly recorded here, so we substitute "GCS = 15" for
   AVPU "Alert" and treat any GCS below 15 as the most severe AVPU band. **Is that
   substitution acceptable, or would you want a real AVPU field sourced instead?**

2. **System correlation layer** -- previously, only one cross-system link was encoded
   by hand (cardiovascular feeding into renal, reflecting the known cardiorenal
   relationship). This is now generalized so the model can learn OTHER cross-system
   relationships too, and reports which systems it found move together most strongly
   in the data. **This is the artifact most worth your review**: once run on real
   data, please sanity-check the top reported correlations against what you'd expect
   clinically -- if cardiorenal shows up strongly, that's a good sign the mechanism is
   working; if something clinically implausible shows up strongly instead, that's
   worth flagging before we trust the model's other outputs.
